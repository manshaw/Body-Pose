QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp
INCLUDEPATH += C:\Opencv\opencv340\build\include
LIBS += C:\Opencv\opencv340-build\bin\libopencv_core340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_highgui340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_imgproc340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_features2d340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_calib3d340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_imgcodecs340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_video340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_videoio340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_videostab340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_stitching340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_objdetect340.dll
LIBS += C:\Opencv\opencv340-build\bin\libopencv_dnn340.dll




