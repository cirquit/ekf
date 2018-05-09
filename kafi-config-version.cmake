# kafi-config-version.cmake - checks version: major must match, minor must be less than or equal

set(PACKAGE_VERSION @KAFI_VERSION@)

if("${PACKAGE_FIND_VERSION_MAJOR}" EQUAL "@KAFI_VERSION_MAJOR@")
    if ("${PACKAGE_FIND_VERSION_MINOR}" EQUAL "@KAFI_VERSION_MINOR@")
        set(PACKAGE_VERSION_EXACT TRUE)
    elseif("${PACKAGE_FIND_VERSION_MINOR}" LESS "@KAFI_VERSION_MINOR@")
        set(PACKAGE_VERSION_COMPATIBLE TRUE)
    else()
        set(PACKAGE_VERSION_UNSUITABLE TRUE)
    endif()
else()
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
endif()