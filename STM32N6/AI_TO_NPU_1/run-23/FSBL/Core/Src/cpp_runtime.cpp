/**
 * cpp_runtime.cpp — minimal embedded C++ runtime for the grammar runner.
 *
 * Provides the replaceable operator new/delete over the newlib heap (malloc/free,
 * backed by _sbrk in sysmem.c, heap = _end .. _estack). This lets the STL
 * containers the grammar runner needs (std::vector / std::map / std::string) work
 * without depending on libstdc++'s throwing operator new.
 *
 * Plus a smoke test that exercises the heap + containers, to validate the C++
 * data-structure infrastructure on device before porting the full runner.
 */
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <map>

/* ── operator new/delete → newlib malloc/free (no-throw, embedded subset) ── */
void* operator new(std::size_t n)               { return malloc(n ? n : 1); }
void* operator new[](std::size_t n)             { return malloc(n ? n : 1); }
void  operator delete(void* p) noexcept         { free(p); }
void  operator delete[](void* p) noexcept       { free(p); }
void  operator delete(void* p, std::size_t) noexcept   { free(p); }
void  operator delete[](void* p, std::size_t) noexcept { free(p); }

/* ── STL smoke test: vector<string> + map<string,int>, exercised on the heap ── */
extern "C" void Cpp_STL_SmokeTest(void)
{
    std::vector<std::string> toks;
    toks.push_back("expr");
    toks.push_back("term");
    toks.push_back("factor");

    std::map<std::string, int> cache;
    for (std::size_t i = 0; i < toks.size(); ++i)
        cache[toks[i]] = (int)toks[i].size();

    std::printf("STL smoke: vec=%u map=%u | ",
                (unsigned)toks.size(), (unsigned)cache.size());
    for (std::map<std::string, int>::iterator it = cache.begin(); it != cache.end(); ++it)
        std::printf("%s:%d ", it->first.c_str(), it->second);
    std::printf("\r\n");
}
