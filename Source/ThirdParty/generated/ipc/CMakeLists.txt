cmake_minimum_required(VERSION 3.10)

#using undocumented cmake functionality to force global on imported targets
function(add_library)
    set(_args ${ARGN})
    if ("${_args}" MATCHES ";IMPORTED")
        list(APPEND _args GLOBAL)
    endif()
    _add_library(${_args})
endfunction()

macro(_ue4_get_target_link_libraries addTarget target_file_list target_linker_file_list)
#    message(STATUS "targets: ${ARGN}")
#    message(STATUS "get_target - addTarget: ${addTarget}  target_file_list: ${target_file_list}  target_linker_file_list: ${target_linker_file_list}")

    foreach(target ${ARGN})
#        message(STATUS "target: ${target}")
        if(TARGET ${target})
            get_target_property(target_libraries ${target} INTERFACE_LINK_LIBRARIES)
#            message(STATUS "target_libraries: ${target_libraries}")

            get_target_property(target_type ${target} TYPE)
#            message(STATUS "target type: ${target_type}")

            if(ON)
#                if(target_type STREQUAL "SHARED_LIBRARY")
                if(NOT target_type STREQUAL "INTERFACE_LIBRARY")
                    set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
#                    message(STATUS "Set MSVC_RUNTIME_LIBRARY: MultiThreadedDLL")
                    get_target_property(isImported ${target} IMPORTED)
                    if(NOT isImported)
                        target_compile_options(${target} PRIVATE "/MD")
                    endif()
                endif()
#                else()
#                    set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
#                    message(STATUS "Set MSVC_RUNTIME_LIBRARY: MultiThreaded")
#                    target_compile_options(${target} PRIVATE "/MT")
#                endif()
            endif()
            
#            get_target_property(${target}_compile_options ${target} COMPILE_OPTIONS)
#            message(STATUS "${target}_compile_options: ${${target}_compile_options}")

            if(target_libraries) 
                #get info for targets libs
                _ue4_get_target_link_libraries(TRUE ${target_file_list} ${target_linker_file_list} ${target_libraries})
            endif()

#            message(STATUS "addTarget: ${addTarget}")
            if(${addTarget})
                if(target_type STREQUAL "SHARED_LIBRARY")
                    list(APPEND ${target_file_list} "\"$<TARGET_FILE:${target}>\"$<COMMA>")
                endif()
                if(target_type STREQUAL "STATIC_LIBRARY"
                    OR target_type STREQUAL "SHARED_LIBRARY")
#                    message(STATUS "adding: ${target}")
                    list(APPEND ${target_linker_file_list} "$<TARGET_LINKER_FILE:${target}>")
                endif ()
            else()
#                message(STATUS "not adding")
            endif()
        else()
            #not a target, need to hunt for library to include
            #currently ignored, use targets instead of files
#            message(STATUS "not a target")

#            get_target_property(target_type ${target} TYPE)
#            message(STATUS "${target} type: ${target_type}")

        endif()
    endforeach()

    if(${target_file_list})
        list(REMOVE_DUPLICATES ${target_file_list})
    endif()
    if(${target_linker_file_list})
        list(REMOVE_DUPLICATES ${target_linker_file_list})
    endif()
endmacro()

macro(ue4_get_target_link_libraries target_file_list target_linker_file_list)
    _ue4_get_target_link_libraries(TRUE ${target_file_list} ${target_linker_file_list} ${ARGN})
endmacro()

macro(target_build_info build_info target)
#    message(STATUS "target_build_info - build_info=${build_info}")
    list(APPEND ${build_info} "includes=$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>>,$<COMMA>>")

    list(APPEND target_file_dependencies "F:/Unreal Projects/alpha-new/unreal-map/Source/UHMP/../Deps/cppipc/CMakeLists.txt")
    list(APPEND target_file_dependencies $<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${target},CMAKE_CONFIGURE_DEPENDS>>)

    list(APPEND target_file_source_dependencies $<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${target},SOURCES>>)
    
#    set(target_file_dependencies "F:/Unreal Projects/alpha-new/unreal-map/Source/UHMP/../Deps/cppipc/CMakeList.txt")
#    set(target_file_dependencies "${target_file_dependencies} $<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${target},SOURCES>>")
#    set(target_file_dependencies "${target_file_dependencies} $<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${target},CMAKE_CONFIGURE_DEPENDS>>")

#    list(APPEND ${build_info} "dependencies=$<JOIN:${target_cmakefile}, $<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${target},SOURCES>>,$<COMMA>>,$<COMMA>>")
    ue4_get_target_link_libraries(target_binary_libs target_link_lib_files ${target})

    string(REPLACE ";" "$<SEMICOLON>" target_file_dependencies "${target_file_dependencies}")
    string(REPLACE ";" "$<SEMICOLON>" target_file_source_dependencies "${target_file_source_dependencies}")
    string(REPLACE ";" "$<SEMICOLON>" target_binary_libs "${target_binary_libs}")
    string(REPLACE ";" "$<SEMICOLON>" target_link_lib_files "${target_link_lib_files}")

#    message(STATUS "target_binary_libs=${target_binary_libs}")
#    message(STATUS "target_link_lib_files=${target_link_lib_files}")

    list(APPEND ${build_info} "dependencies=$<JOIN:${target_file_dependencies},$<COMMA>>")
    list(APPEND ${build_info} "sourcePath=$<TARGET_PROPERTY:${target},SOURCE_DIR>")
    list(APPEND ${build_info} "sourceDependencies=$<JOIN:${target_file_source_dependencies},$<COMMA>>")
    list(APPEND ${build_info} "binaries=$<JOIN:${target_binary_libs},$<COMMA>>")
    list(APPEND ${build_info} "libraries=$<JOIN:${target_link_lib_files},$<COMMA>>")
endmacro()

cmake_policy(SET CMP0091 NEW)

set(ipc_build_info "")

#if(ON)
#    message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
#    string(REPLACE "/MDd" "/MD" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
#    string(REPLACE "/MTd" "/MT" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
#    message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
#endif()

add_subdirectory("F:/Unreal Projects/alpha-new/unreal-map/Source/UHMP/../Deps/cppipc" "F:/Unreal Projects/alpha-new/unreal-map/Source/UHMP/../ThirdParty/generated/ipc/build/ipc")
target_build_info("ipc_build_info" ipc)

project(ipc_build)

string(REPLACE ";" "\n" ipc_build_info "${ipc_build_info}")
#message(STATUS "build_info=${ipc_build_info}")

#file(WRITE "F:/Unreal Projects/alpha-new/unreal-map/Source/UHMP/../ThirdParty/generated/ipc/build/buildinfo_$<CONFIG>.template" ${ipc_build_info})
file(GENERATE OUTPUT "F:/Unreal Projects/alpha-new/unreal-map/Source/UHMP/../ThirdParty/generated/ipc/build/buildinfo_$<CONFIG>.output" CONTENT "${ipc_build_info}")