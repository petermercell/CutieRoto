// ============================================================================
// test_cache_race.cpp — standalone sanitizer reproduction + regression test for
// the CutieRoto matte-cache serve/clear race. NO Nuke, NO libtorch — pure std.
//
// It models the exact pattern from the plugin:
//   - a serve  mimics CutieRoto::engine()      (look up a frame, then read it)
//   - a writer mimics processAllFrames()        (re-fill frames, erase, clear())
//
//   FIXED build (default, shared_ptr cache) -> sanitizer stays SILENT  => PASS
//   BUGGY build (-DCACHE_BUGGY, raw pointer) -> sanitizer REPORTS       => proves
//                                               the test actually has teeth
//
// The fixed path copies a shared_ptr<const Matte> under the lock and serves with
// the lock released; the buffer stays alive via that reference even if a writer
// erases the entry or clears the whole map. The buggy path keeps a RAW pointer
// into the map-owned vector after unlocking — a concurrent erase()/clear() frees
// it under the reader.
//
// TWO PHASES, because the two sanitizers detect different facets of the bug:
//
//   1) deterministicProbe()  — forces the free-then-read interleaving with a
//      handshake, with NO reallocation in between, so the freed buffer is still
//      poisoned when read. This makes AddressSanitizer report a use-after-free
//      EVERY run (no timing luck). It is ordered (acquire/release), so it is a
//      clean UAF rather than a data race — ThreadSanitizer will NOT flag it.
//      This is the reliable gate; prefer -DCUTIE_RACE_SANITIZER=address.
//
//   2) stressLoop()          — many readers + a writer hammering concurrently
//      with a widened read window, exercising the genuine *data race* facet that
//      ThreadSanitizer detects. Timing-dependent by nature, but with the wide
//      window it trips TSan reliably over the run.
//
// Build via CMake: -DCUTIE_SANITIZE_RACE=ON (see CMakeLists). Run both targets:
//   ./build/test_cache_race          # expect: clean, no sanitizer output
//   ./build/test_cache_race_buggy    # expect: sanitizer report (UAF / race)
// ============================================================================
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace {

// Same SpinLock the plugin uses. acquire/release give the sanitizer a proper
// happens-before edge, so it won't false-positive on the lock itself — only on
// the genuinely unsynchronized buffer access in the buggy path.
struct SpinLock {
    std::atomic_flag f = ATOMIC_FLAG_INIT;
    void lock()   { while (f.test_and_set(std::memory_order_acquire)) {} }
    void unlock() { f.clear(std::memory_order_release); }
};
struct SpinGuard {
    SpinLock& s;
    explicit SpinGuard(SpinLock& l) : s(l) { s.lock(); }
    ~SpinGuard() { s.unlock(); }
};

struct Matte { std::vector<float> a; int w = 0, h = 0; };

constexpr int  MATTE_W = 256, MATTE_H = 256;   // big enough that a serve overlaps a free
constexpr int  NUM_FRAMES  = 24;               // distinct cache keys
constexpr int  NUM_READERS = 8;                // mimic Nuke worker threads
constexpr auto RUN_FOR     = std::chrono::milliseconds(1500);

#ifdef CACHE_BUGGY
using Cache = std::map<int, Matte>;                       // map OWNS the vector
#else
using MattePtr = std::shared_ptr<const Matte>;            // immutable, ref-counted
using Cache    = std::map<int, MattePtr>;
#endif

Cache    g_cache;
SpinLock g_lock;
std::atomic<bool>     g_stop{false};
std::atomic<uint64_t> g_serves{0};
std::vector<double>   g_localSums(NUM_READERS, 0.0);      // per-thread; summed after join

std::shared_ptr<Matte> makeMatte(int frame) {
    auto m = std::make_shared<Matte>();
    m->w = MATTE_W; m->h = MATTE_H;
    m->a.assign((size_t)MATTE_W * MATTE_H, (float)(frame % 7) * 0.125f);
    return m;
}

void put(int frame, const std::shared_ptr<Matte>& m) {
#ifdef CACHE_BUGGY
    g_cache[frame] = *m;     // by value into the map
#else
    g_cache[frame] = m;      // shared_ptr into the map
#endif
}

// ----------------------------------------------------------------------------
// PHASE 1 — deterministic free-then-read. Reliable use-after-free for ASan.
// ----------------------------------------------------------------------------
double deterministicProbe() {
    const int F = 0;
    { SpinGuard lk(g_lock); put(F, makeMatte(F)); }

    std::atomic<int> phase{0};   // 0 init -> 1 reader holds handle -> 2 freed
    std::thread freer([&]{
        while (phase.load(std::memory_order_acquire) != 1) std::this_thread::yield();
        { SpinGuard lk(g_lock); g_cache.erase(F); }   // frees the buffer (buggy) / drops map ref (fixed)
        phase.store(2, std::memory_order_release);
    });

    double sum = 0.0;
#ifdef CACHE_BUGGY
    const float* base = nullptr; int w = 0, h = 0;
    {
        SpinGuard lk(g_lock);
        auto it = g_cache.find(F);
        base = it->second.a.data();           // raw pointer into the map-owned vector
        w = it->second.w; h = it->second.h;
    }
    phase.store(1, std::memory_order_release);                                    // "I hold the pointer"
    while (phase.load(std::memory_order_acquire) != 2) std::this_thread::yield(); // wait until it's freed
    for (int y = 0; y < h; ++y) {              // <-- read of freed, still-poisoned memory: ASan UAF
        const float* row = base + (size_t)y * w;
        for (int x = 0; x < w; ++x) sum += row[x];
    }
#else
    MattePtr cached;
    {
        SpinGuard lk(g_lock);
        auto it = g_cache.find(F);
        cached = it->second;                   // our own ref; survives the erase below
    }
    phase.store(1, std::memory_order_release);
    while (phase.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    for (int y = 0; y < cached->h; ++y) {      // safe: shared_ptr kept the buffer alive
        const float* row = cached->a.data() + (size_t)y * cached->w;
        for (int x = 0; x < cached->w; ++x) sum += row[x];
    }
#endif
    freer.join();
    return sum;
}

// ----------------------------------------------------------------------------
// PHASE 2 — concurrent stress. Exercises the data-race facet for TSan.
// ----------------------------------------------------------------------------
double serveOnce(int frame) {
#ifdef CACHE_BUGGY
    const float* base = nullptr; int w = 0, h = 0;
    {
        SpinGuard lk(g_lock);
        auto it = g_cache.find(frame);
        if (it == g_cache.end()) return 0.0;
        base = it->second.a.data();            // BUG: raw pointer into map-owned vector
        w = it->second.w; h = it->second.h;
    } // <-- lock released; a writer may now erase/clear and FREE this buffer
    std::this_thread::sleep_for(std::chrono::microseconds(30));  // widen the window
    double sum = 0.0;
    for (int y = 0; y < h; ++y) {
        const float* row = base + (size_t)y * w;
        for (int x = 0; x < w; ++x) sum += row[x];   // <-- use-after-free / data-race window
    }
    return sum;
#else
    MattePtr cached;
    {
        SpinGuard lk(g_lock);
        auto it = g_cache.find(frame);
        if (it != g_cache.end()) cached = it->second;  // ref-count bump under lock
    } // lock released; our shared_ptr keeps the buffer alive no matter what
    if (!cached) return 0.0;
    std::this_thread::sleep_for(std::chrono::microseconds(30));
    double sum = 0.0;
    for (int y = 0; y < cached->h; ++y) {
        const float* row = cached->a.data() + (size_t)y * cached->w;
        for (int x = 0; x < cached->w; ++x) sum += row[x];
    }
    return sum;
#endif
}

void reader(int idx, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, NUM_FRAMES - 1);
    double local = 0.0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        local += serveOnce(pick(rng));
        g_serves.fetch_add(1, std::memory_order_relaxed);
    }
    g_localSums[idx] = local;   // only this thread writes its slot
}

// Mimics processAllFrames(): keep frames populated while constantly evicting and
// periodically clear()ing — maximizing the chance a reader is mid-read on a
// buffer that's being freed.
void writer() {
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int> pick(0, NUM_FRAMES - 1);
    int tick = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        const int f = pick(rng);
        {
            SpinGuard lk(g_lock);
            put(f, makeMatte(f));
            g_cache.erase((f + NUM_FRAMES / 2) % NUM_FRAMES);  // free a different populated entry
        }
        if ((++tick % 32) == 0) {            // a "Process" run: wipe everything
            SpinGuard lk(g_lock);
            g_cache.clear();
        }
        std::this_thread::yield();
    }
}

void stressLoop() {
    {   // seed all frames
        SpinGuard lk(g_lock);
        for (int f = 0; f < NUM_FRAMES; ++f) put(f, makeMatte(f));
    }
    std::vector<std::thread> ts;
    ts.emplace_back(writer);
    for (int i = 0; i < NUM_READERS; ++i) ts.emplace_back(reader, i, 1234u + (unsigned)i);
    std::this_thread::sleep_for(RUN_FOR);
    g_stop.store(true, std::memory_order_relaxed);
    for (auto& t : ts) t.join();
}

} // namespace

int main() {
#ifdef CACHE_BUGGY
    std::printf("[test_cache_race] BUGGY build (raw pointer) — sanitizer SHOULD report a UAF/race\n");
#else
    std::printf("[test_cache_race] FIXED build (shared_ptr) — sanitizer should stay SILENT\n");
#endif

    std::printf("[test_cache_race] phase 1: deterministic free-then-read probe...\n");
    double probeSum = deterministicProbe();   // ASan halts here on the buggy build

    std::printf("[test_cache_race] phase 2: concurrent stress (%d readers)...\n", NUM_READERS);
    stressLoop();

    double sink = probeSum;
    for (double d : g_localSums) sink += d;
    std::printf("[test_cache_race] done: %llu serves, sink=%g\n",
                (unsigned long long)g_serves.load(), sink);
    std::printf("[test_cache_race] no sanitizer report above => this build is clean.\n");
    return 0;
}
