/*
 * Copyright (c) 2020 Andreas Pohl
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * Author: Andreas Pohl
 */

#include "Screen.h"

#ifdef JUCE_WINDOWS
#include <windows.h>

namespace e47 {

std::shared_ptr<juce::Image> captureScreenNative(juce::Rectangle<int> rect) {
    HDC hDC = GetDC(0);
    float dpi = (GetDeviceCaps(hDC, LOGPIXELSX) + GetDeviceCaps(hDC, LOGPIXELSY)) / 2.0f;
    float scaleFactor = dpi / 96;
    int x = GetSystemMetrics(SM_XVIRTUALSCREEN) + rect.getX();
    int y = GetSystemMetrics(SM_YVIRTUALSCREEN) + rect.getY();
    int w = (int)roundl(rect.getWidth() * scaleFactor);
    int h = (int)roundl(rect.getHeight() * scaleFactor);
    HDC cDC = CreateCompatibleDC(hDC);
    HBITMAP bmap = CreateCompatibleBitmap(hDC, w, h);
    HGDIOBJ oldObj = SelectObject(cDC, bmap);
    auto ret = std::make_shared<juce::Image>(juce::Image::ARGB, w, h, false);

    if (BitBlt(cDC, 0, 0, w, h, hDC, x, y, SRCCOPY)) {
        BITMAPINFO bmi = {0};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = w;
        bmi.bmiHeader.biHeight = -h;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        bmi.bmiHeader.biSizeImage = 0;

        juce::Image::BitmapData bd(*ret, 0, 0, w, h);

        if (!GetDIBits(hDC, bmap, 0, h, bd.data, &bmi, DIB_RGB_COLORS)) {
            ret.reset();
        }
    }

    SelectObject(cDC, oldObj);
    DeleteObject(bmap);
    DeleteDC(cDC);
    ReleaseDC(0, hDC);

    if (scaleFactor != 1.0) {
        auto rescaled = std::make_shared<juce::Image>(ret->rescaled(rect.getWidth(), rect.getHeight()));
        return rescaled;
    }
    return ret;
}

}  // namespace e47

#endif

#if defined(JUCE_LINUX)
namespace e47 {

int getScreenShotData(unsigned char** buffer, int x, int y, unsigned int w, unsigned int h);

std::shared_ptr<juce::Image> captureScreenNative(juce::Rectangle<int> rect) {
    unsigned int w = (unsigned int)rect.getWidth();
    unsigned int h = (unsigned int)rect.getHeight();
    int x = rect.getX(), y = rect.getY();
    // ALOCATE buffer for copying image data
    auto ret = std::make_shared<juce::Image>(juce::Image::ARGB, w, h, false);
    juce::Image::BitmapData bd(*ret, 0, 0, (int)w, (int)h);

    /* GET Image */
    getScreenShotData(&bd.data, x, y, w, h);
    return ret;
}

}  // namespace e47
#endif
