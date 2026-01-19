# CMakeList.txt : CppMlpack の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#
cmake_minimum_required (VERSION 3.8)

# サポートされている場合は、MSVC コンパイラのホット リロードを有効にします。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CppMlpack")

# vcpkgツールチェインファイルが設定されていない場合、環境変数から取得を試みる
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# vcpkg_installedディレクトリを検索パスに追加
if(EXISTS "${CMAKE_BINARY_DIR}/vcpkg_installed")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share")
    # インクルードパスも追加
    list(APPEND CMAKE_INCLUDE_PATH "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include")
endif()

# ソースをこのプロジェクトの実行可能ファイルに追加します。
add_executable (CppMlpack "main.cpp" "csv_loader.hpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CppMlpack PROPERTY CXX_STANDARD 20)
endif()

# mlpackを検索（ヘッダーのみライブラリなのでfind_pathを使用）
find_path(MLPACK_INCLUDE_DIRS "mlpack.hpp")
if(NOT MLPACK_INCLUDE_DIRS)
    message(FATAL_ERROR "mlpackが見つかりません。vcpkgでインストールしてください。")
endif()
target_include_directories(CppMlpack PRIVATE ${MLPACK_INCLUDE_DIRS})

# Armadilloを検索（mlpackの依存関係）
find_package(Armadillo CONFIG REQUIRED)

# nlohmann-jsonを検索
find_package(nlohmann_json CONFIG REQUIRED)

# csv-parserを検索（vcpkgの推奨方法）
find_package(unofficial-vincentlaucsb-csv-parser CONFIG REQUIRED)

# ライブラリをリンク
target_link_libraries(CppMlpack PRIVATE
    armadillo
    nlohmann_json::nlohmann_json
    unofficial::vincentlaucsb-csv-parser::csv
)

# 数学ライブラリをリンク（M_PIなどに必要）
if(UNIX)
    target_link_libraries(CppMlpack PRIVATE m)
endif()
