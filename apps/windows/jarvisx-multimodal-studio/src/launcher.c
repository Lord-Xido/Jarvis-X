// JARVIS-X Multimodal Studio Windows launcher.
// Builds without a C runtime and extracts the embedded PowerShell runtime and HTML UI.

typedef void* HANDLE;
typedef void* LPVOID;
typedef const void* LPCVOID;
typedef unsigned long DWORD;
typedef int BOOL;
typedef unsigned short WORD;
typedef unsigned long long SIZE_T;
typedef unsigned short WCHAR;
typedef WCHAR* LPWSTR;
typedef const WCHAR* LPCWSTR;

#define WINAPI __stdcall
#define NULL ((void*)0)
#define INVALID_HANDLE_VALUE ((HANDLE)(long long)-1)

#define GENERIC_WRITE 0x40000000UL
#define FILE_SHARE_READ 0x00000001UL
#define CREATE_ALWAYS 2UL
#define FILE_ATTRIBUTE_NORMAL 0x00000080UL
#define HEAP_ZERO_MEMORY 0x00000008UL
#define CREATE_NO_WINDOW 0x08000000UL
#define STARTF_USESHOWWINDOW 0x00000001UL
#define SW_HIDE 0
#define MB_OK 0x00000000UL
#define MB_ICONERROR 0x00000010UL

typedef struct _STARTUPINFOW {
    DWORD cb;
    LPWSTR lpReserved;
    LPWSTR lpDesktop;
    LPWSTR lpTitle;
    DWORD dwX;
    DWORD dwY;
    DWORD dwXSize;
    DWORD dwYSize;
    DWORD dwXCountChars;
    DWORD dwYCountChars;
    DWORD dwFillAttribute;
    DWORD dwFlags;
    WORD wShowWindow;
    WORD cbReserved2;
    unsigned char* lpReserved2;
    HANDLE hStdInput;
    HANDLE hStdOutput;
    HANDLE hStdError;
} STARTUPINFOW;

typedef struct _PROCESS_INFORMATION {
    HANDLE hProcess;
    HANDLE hThread;
    DWORD dwProcessId;
    DWORD dwThreadId;
} PROCESS_INFORMATION;

__declspec(dllimport) DWORD WINAPI GetEnvironmentVariableW(LPCWSTR, LPWSTR, DWORD);
__declspec(dllimport) BOOL WINAPI CreateDirectoryW(LPCWSTR, LPVOID);
__declspec(dllimport) HANDLE WINAPI CreateFileW(LPCWSTR, DWORD, DWORD, LPVOID, DWORD, DWORD, HANDLE);
__declspec(dllimport) BOOL WINAPI WriteFile(HANDLE, LPCVOID, DWORD, DWORD*, LPVOID);
__declspec(dllimport) BOOL WINAPI CloseHandle(HANDLE);
__declspec(dllimport) HANDLE WINAPI GetProcessHeap(void);
__declspec(dllimport) LPVOID WINAPI HeapAlloc(HANDLE, DWORD, SIZE_T);
__declspec(dllimport) BOOL WINAPI HeapFree(HANDLE, DWORD, LPVOID);
__declspec(dllimport) BOOL WINAPI CreateProcessW(
    LPCWSTR, LPWSTR, LPVOID, LPVOID, BOOL, DWORD, LPVOID, LPCWSTR,
    STARTUPINFOW*, PROCESS_INFORMATION*
);
__declspec(dllimport) void WINAPI ExitProcess(DWORD);
__declspec(dllimport) int WINAPI MessageBoxW(void*, LPCWSTR, LPCWSTR, unsigned int);

#include "embedded_assets.inc"

static SIZE_T wlen(const WCHAR* s) {
    SIZE_T n = 0;
    while (s && s[n]) n++;
    return n;
}

static void wcopy(WCHAR* dst, const WCHAR* src) {
    while ((*dst++ = *src++) != 0) {}
}

static void wcat(WCHAR* dst, const WCHAR* src) {
    SIZE_T n = wlen(dst);
    wcopy(dst + n, src);
}

static void zero_mem(void* ptr, SIZE_T bytes) {
    unsigned char* p = (unsigned char*)ptr;
    SIZE_T i;
    for (i = 0; i < bytes; i++) p[i] = 0;
}

static BOOL write_asset(const WCHAR* path, const unsigned char* data, DWORD size) {
    HANDLE file = CreateFileW(
        path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, NULL
    );
    DWORD written = 0;
    BOOL ok;
    if (file == INVALID_HANDLE_VALUE) return 0;
    ok = WriteFile(file, data, size, &written, NULL);
    CloseHandle(file);
    return ok && written == size;
}

static void fail(const WCHAR* message, DWORD code) {
    MessageBoxW(NULL, message, L"JARVIS-X Multimodal Studio", MB_OK | MB_ICONERROR);
    ExitProcess(code);
}

void wWinMainCRTStartup(void) {
    HANDLE heap = GetProcessHeap();
    WCHAR* appDir = (WCHAR*)HeapAlloc(heap, HEAP_ZERO_MEMORY, 8192 * sizeof(WCHAR));
    WCHAR* scriptPath = (WCHAR*)HeapAlloc(heap, HEAP_ZERO_MEMORY, 8192 * sizeof(WCHAR));
    WCHAR* htmlPath = (WCHAR*)HeapAlloc(heap, HEAP_ZERO_MEMORY, 8192 * sizeof(WCHAR));
    WCHAR* commandLine = (WCHAR*)HeapAlloc(heap, HEAP_ZERO_MEMORY, 16384 * sizeof(WCHAR));
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    DWORD length;
    BOOL started;

    if (!appDir || !scriptPath || !htmlPath || !commandLine) {
        fail(L"Unable to allocate launcher memory.", 10);
    }

    length = GetEnvironmentVariableW(L"LOCALAPPDATA", appDir, 8190);
    if (length == 0 || length >= 8190) {
        length = GetEnvironmentVariableW(L"USERPROFILE", appDir, 8150);
        if (length == 0 || length >= 8150) {
            fail(L"Unable to resolve the Windows user application-data folder.", 11);
        }
        wcat(appDir, L"\\AppData\\Local");
    }

    wcat(appDir, L"\\JarvisXMultimodal");
    CreateDirectoryW(appDir, NULL);

    wcopy(scriptPath, appDir);
    wcat(scriptPath, L"\\JarvisXMultimodal.ps1");
    wcopy(htmlPath, appDir);
    wcat(htmlPath, L"\\index.html");

    if (!write_asset(scriptPath, APP_PS1, APP_PS1_len)) {
        fail(L"Unable to install the local JARVIS-X runtime script.", 12);
    }
    if (!write_asset(htmlPath, APP_HTML, APP_HTML_len)) {
        fail(L"Unable to install the JARVIS-X graphical interface.", 13);
    }

    wcopy(commandLine, L"powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"");
    wcat(commandLine, scriptPath);
    wcat(commandLine, L"\"");

    zero_mem(&si, sizeof(si));
    zero_mem(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    started = CreateProcessW(
        NULL, commandLine, NULL, NULL, 0, CREATE_NO_WINDOW, NULL, appDir, &si, &pi
    );

    if (!started) {
        fail(
            L"Unable to start Windows PowerShell. Windows PowerShell 5.1 or later is required.",
            14
        );
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    HeapFree(heap, 0, commandLine);
    HeapFree(heap, 0, htmlPath);
    HeapFree(heap, 0, scriptPath);
    HeapFree(heap, 0, appDir);
    ExitProcess(0);
}
