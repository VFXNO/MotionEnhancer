Backup taken before adding dominant-motion candidates and FFX brightness matching (2026-10-14).

run\    - the working program as it was (motion_enhancer.exe + compiled shaders). Run it from this folder.
source\ - shaders, src, include, CMakeLists.txt, README.md at that point.

To restore the source: copy the contents of source\ over the project folder, then rebuild:
    cmake --build build-ffx-local --config Release
