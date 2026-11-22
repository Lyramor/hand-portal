#!/bin/zsh
# Membuat Hand Portal.app dari sumbernya.
#
# Bundle ini perlu karena macOS hanya memberi izin kamera kepada aplikasi yang
# mendeklarasikan NSCameraUsageDescription di Info.plist. Menjalankan Python
# langsung dari Terminal tidak punya deklarasi itu, jadi permintaan kameranya
# ditolak tanpa dialog.
#
# Pakai: ./tools/build_app.sh

set -e
cd "$(dirname "$0")/.."

APP="Hand Portal.app"
BIN="$APP/Contents/MacOS/HandPortal"

mkdir -p "$APP/Contents/MacOS"
cp tools/AppInfo.plist "$APP/Contents/Info.plist"

clang -O2 -Wall -o "$BIN" tools/launcher.c

# Tanda tangan adhoc sudah cukup untuk dipakai sendiri. Tanpa ini macOS
# menganggap bundle berubah dan izin kameranya harus diberikan ulang.
codesign --force --sign - "$APP"

echo "Selesai: $APP"
echo "Buka dengan: open \"$APP\""
