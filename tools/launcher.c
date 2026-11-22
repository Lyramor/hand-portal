// Launcher tipis untuk app bundle.
//
// LaunchServices di macOS modern tidak mau menjalankan app bundle yang
// executable utamanya berupa skrip shell, jadi dibutuhkan binary Mach-O
// sungguhan. Binary ini hanya mencari folder proyek lalu menjalankan
// run.sh di dalam Terminal? Tidak: dia menjalankan python langsung, supaya
// prosesnya tetap berada di bawah app bundle ini dan izin kameranya mengikuti
// bundle, bukan mengikuti Terminal.
//
// Argumen dari Finder (misalnya --probe-camera) diteruskan apa adanya.

#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Bundle ada di <proyek>/Hand Portal.app/Contents/MacOS/HandPortal, jadi
// folder proyek tiga tingkat di atas binary ini.
static int project_dir(char *out, size_t size) {
    char executable[PATH_MAX];
    uint32_t length = sizeof(executable);
    if (_NSGetExecutablePath(executable, &length) != 0) {
        return 0;
    }
    char resolved[PATH_MAX];
    if (realpath(executable, resolved) == NULL) {
        return 0;
    }
    char *slash = strrchr(resolved, '/');
    if (slash == NULL) {
        return 0;
    }
    *slash = '\0';  // .../Contents/MacOS
    for (int i = 0; i < 3; i++) {
        slash = strrchr(resolved, '/');
        if (slash == NULL) {
            return 0;
        }
        *slash = '\0';
    }
    snprintf(out, size, "%s", resolved);
    return 1;
}

int main(int argc, char *argv[]) {
    char dir[PATH_MAX];
    if (!project_dir(dir, sizeof(dir))) {
        fprintf(stderr, "Tidak bisa menemukan folder proyek dari lokasi aplikasi.\n");
        return 1;
    }

    char python[PATH_MAX];
    snprintf(python, sizeof(python), "%s/.venv/bin/python", dir);
    if (access(python, X_OK) != 0) {
        fprintf(stderr, "Python di %s belum siap. Jalankan ./run.sh sekali dulu.\n", python);
        return 1;
    }

    if (chdir(dir) != 0) {
        perror("chdir");
        return 1;
    }

    // App yang dibuka dari Finder tidak punya terminal, jadi keluaran program
    // diarahkan ke berkas log. Tanpa ini pesan galat hilang tanpa jejak.
    char log_path[PATH_MAX];
    const char *home = getenv("HOME");
    if (home != NULL) {
        snprintf(log_path, sizeof(log_path), "%s/Library/Logs/HandPortal.log", home);
        freopen(log_path, "a", stdout);
        freopen(log_path, "a", stderr);
    }

    char script[PATH_MAX];
    snprintf(script, sizeof(script), "%s/main.py", dir);

    // Argumen disusun ulang: python, main.py, lalu argumen dari pemanggil.
    char **args = calloc((size_t)argc + 2, sizeof(char *));
    if (args == NULL) {
        return 1;
    }
    args[0] = python;
    args[1] = script;
    for (int i = 1; i < argc; i++) {
        args[i + 1] = argv[i];
    }
    args[argc + 1] = NULL;

    // Python dijalankan sebagai proses anak, bukan menggantikan launcher.
    // Alasannya soal izin: TCC menilai "responsible process" dari aplikasi
    // induk. Kalau launcher ini di-execv jadi python, aplikasi induknya hilang
    // dan permintaan kamera ditolak tanpa dialog karena python tidak punya
    // deklarasi izin. Dengan fork, induknya tetap app bundle ini.
    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return 1;
    }
    if (child == 0) {
        execv(python, args);
        perror("execv");
        _exit(127);
    }

    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
        perror("waitpid");
        return 1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 1;
}
