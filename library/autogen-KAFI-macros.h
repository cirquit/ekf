
#ifndef AUTOGEN_KAFI_MACROS_H
#define AUTOGEN_KAFI_MACROS_H
    #include <string.h>
    //! macro preprocessing of the file standart to get the filename instead of the full filepath
    #define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

    #include <iostream>
    //! logging based on the DEBUG_LEVEL_KAFI defined while building with cmake (only active with DEBUG_LEVEL=2)
    #define DEBUG_MSG_KAFI(msg) std::cerr << "[KAFI - " \
                        << __FILENAME__ << ':' \
                        << __LINE__ << ':'     \
                        << __func__ << "()]: "    \
                        << msg;
    //! logging based on the DEBUG_LEVEL_KAFI defined while building with cmake (active with DEBUG_LEVEL=1 and 2)
    #define DEBUG_CRIT_MSG_KAFI(msg) DEBUG_MSG_KAFI(msg)
#endif // AUTOGEN_KAFI_MACROS_H