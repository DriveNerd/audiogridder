cmake_minimum_required(VERSION 3.15)

project(AUDIOGRIDDER_SERVER VERSION 1.0.0)

set(AG_ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icon.png")
set(AG_ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icon.png")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(AG_ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icon64.png")
  set(AG_ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icon16.png")
endif()

juce_add_gui_app(AudioGridderServer
  VERSION "${AG_VERSION}"
  PRODUCT_NAME "AudioGridderServer"
  COMPANY_NAME "e47"
  COMPANY_COPYRIGHT "2020-2022 Andreas Pohl"
  COMPANY_WEBSITE "https://www.audiogridder.com"
  PLUGINHOST_AU TRUE
  ICON_BIG "${AG_ICON_BIG}"
  ICON_SMALL "${AG_ICON_SMALL}")

juce_generate_juce_header(AudioGridderServer)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/Source AG_SOURCES_SERVER)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/Source/ServerSettings AG_SOURCES_SERVER_SETTINGS)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  list(REMOVE_ITEM AG_SOURCES_SERVER "${CMAKE_CURRENT_SOURCE_DIR}/Source/Screen.mm")
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(REMOVE_ITEM  AG_SOURCES_SERVER "${CMAKE_CURRENT_SOURCE_DIR}/Source/ScreenHelper_linux.cpp")
endif()

# Add GPU sources if CUDA is enabled
set(AG_GPU_SOURCES "")
set(AG_CUDA_SOURCES "")
if(AG_ENABLE_CUDA AND CUDAToolkit_FOUND)
  list(APPEND AG_GPU_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/CUDAManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/GPUAudioBuffer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/GPUProcessor.cpp")
  list(APPEND AG_CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/AudioKernels.cu")
  message(STATUS "CUDA enabled - GPU acceleration available")
else()
  message(STATUS "CUDA disabled - removing GPU sources from build")
  # Remove GPU source files from the main source list to prevent compilation
  list(REMOVE_ITEM AG_SOURCES_SERVER 
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/CUDAManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/GPUAudioBuffer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/GPUProcessor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/AudioKernels.cu")
endif()

target_sources(AudioGridderServer PRIVATE ${AG_SOURCES_SERVER} ${AG_SOURCES_SERVER_SETTINGS} ${AG_SOURCES_COMMON} ${AG_GPU_SOURCES} ${AG_CUDA_SOURCES})

# Set CUDA properties for .cu files
if(AG_ENABLE_CUDA AND CUDAToolkit_FOUND)
  set_property(TARGET AudioGridderServer PROPERTY CUDA_STANDARD 17)
  set_property(TARGET AudioGridderServer PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()

set(AG_PLUGINHOST_VST 0)
if(AG_SDKS_ROOT)
  set(AG_PLUGINHOST_VST 1)
endif()

# Set CUDA compile definitions
if(AG_ENABLE_CUDA AND CUDAToolkit_FOUND)
  set(AG_CUDA_ENABLED 1)
else()
  set(AG_CUDA_ENABLED 0)
endif()

target_compile_definitions(AudioGridderServer PRIVATE
  AG_SERVER
  AG_ENABLE_CUDA=${AG_CUDA_ENABLED}
  AG_SENTRY_ENABLED=${AG_SENTRY_ENABLED}
  AG_SENTRY_DSN="${AG_SENTRY_DSN}"
  AG_SENTRY_CRASHPAD_PATH="${AG_SENTRY_CRASHPAD_PATH}"
  JUCE_PLUGINHOST_VST3=1
  JUCE_PLUGINHOST_VST=${AG_PLUGINHOST_VST}
  JUCE_PLUGINHOST_LV2=1
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  JUCE_MODAL_LOOPS_PERMITTED=1
  JUCE_DISABLE_AUDIOPROCESSOR_BEGIN_END_GESTURE_CHECKING
  JUCE_DISABLE_ASSERTIONS
  JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:AudioGridderServer,JUCE_PRODUCT_NAME>"
  JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:AudioGridderServer,JUCE_VERSION>"
  JUCE_DISPLAY_SPLASH_SCREEN=0)

target_compile_features(AudioGridderServer PRIVATE cxx_std_14)

juce_add_binary_data(AudioGridderServerData SOURCES
  Resources/icon.png
  Resources/icon16.png
  Resources/icon64.png)

target_link_libraries(AudioGridderServer PRIVATE
  AudioGridderServerData
  juce::juce_audio_basics
  juce::juce_audio_processors
  juce::juce_audio_formats
  juce::juce_graphics
  juce::juce_gui_extra
  juce::juce_recommended_config_flags
  juce::juce_recommended_lto_flags
  juce::juce_recommended_warning_flags)

# Link optional dependencies conditionally
if(AG_HAVE_FFMPEG AND FFMPEG_LIBRARIES)
  target_link_libraries(AudioGridderServer PRIVATE ${FFMPEG_LIBRARIES})
  target_compile_definitions(AudioGridderServer PRIVATE AG_FFMPEG=1)
endif()

if(AG_HAVE_WEBP AND WEBP_LIBRARIES)
  target_link_libraries(AudioGridderServer PRIVATE ${WEBP_LIBRARIES})
  target_compile_definitions(AudioGridderServer PRIVATE AG_WEBP=1)
endif()

if(SENTRY_LIBRARIES)
  target_link_libraries(AudioGridderServer PRIVATE ${SENTRY_LIBRARIES})
endif()

# Link CUDA libraries if enabled
if(AG_ENABLE_CUDA AND CUDAToolkit_FOUND)
  target_link_libraries(AudioGridderServer PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(AudioGridderServer PRIVATE  X11 Xtst xcb xcb-shm xcb-xfixes xcb-shape)
  target_compile_definitions(AudioGridderServer PRIVATE JUCE_LINUX=1)
endif()

ag_strip(AudioGridderServer Server)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_libraries(AudioGridderServer
          PRIVATE
          "-framework AVFoundation"
          "-framework CoreMedia"
          "-framework OpenGL"
          "-framework Security")
  if(AG_MACOS_TARGET STRGREATER_EQUAL 10.8)
    target_link_libraries(AudioGridderServer PRIVATE "-framework VideoToolbox")
  endif()
  ag_bundle_add_crashpad(AudioGridderServer)
  ag_bundle_sign(AudioGridderServer)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(AG_ENABLE_ASAN)
      target_compile_options(AudioGridderServer PRIVATE -fsanitize=address -fno-omit-frame-pointer)
      target_link_options(AudioGridderServer PRIVATE -fsanitize=address)
    endif()
    if(AG_ENABLE_DEBUG_COPY_STEP)
      ag_bundle_copy(AudioGridderServer /Applications/Debug)
    endif()
  endif()
endif()
