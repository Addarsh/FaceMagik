//
//  CIImageHelper.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/11/21.
//

import UIKit

class CIImageHelper {
    
    // overlayMask returns an image that overlays given mask on given image.
    static func overlayMask(image: CIImage, mask: CIImage) -> CIImage {
        let blend = CIFilter.blendWithMask()
        blend.backgroundImage = image
        blend.inputImage = CIImage(color: .green)
        blend.maskImage =  mask
        return blend.outputImage!
    }
    
    // bitwiseAnd returns a mask that applies the bitwise AND operation on given masks.
    static func bitwiseAnd(firstMask: CIImage?, secondMask: CIImage?) -> CIImage? {
        let comp = CIFilter.minimumCompositing()
        comp.inputImage = firstMask
        comp.backgroundImage = secondMask
        return comp.outputImage
    }
    
    // bitwiseAnd returns a mask that applies the bitwise XOR operation on given masks.
    static func bitwiseXor(firstMask: CIImage?, secondMask: CIImage?) -> CIImage? {
        let comp = CIFilter.differenceBlendMode()
        comp.backgroundImage = firstMask
        comp.inputImage = secondMask
        return comp.outputImage
    }
    
    // ValueImage converts given sRGB image into Value (in HSV) image.
    static func valueImage(image: CIImage) -> CIImage {
        let gray = CIFilter.maximumComponent()
        gray.inputImage = image
        return gray.outputImage!
    }
    
    // averageValue returns average Value (from HSV) of given sRGB within given mask bounding box.
    static func averageValue(rgbImage: CIImage, maskBbox: CGRect) -> UInt8? {
        guard let avg = CIFilter(name: "CIAreaAverage", parameters: ["inputImage": CIImageHelper.valueImage(image: rgbImage), "inputExtent": CIVector(cgRect: maskBbox)]) else {
            print ("Could not construct CIFilter average")
            return nil
        }
        guard let output = avg.outputImage else {
            print ("averageValue: outputimage nil")
            return nil
        }
        return CIImageHelper.valueAtIndex(output: output, index: 0)
    }
    
    // valueAtIndex returns the Uint8 value at given index of a CIImage that is
    // an output of a filter operation. Use it only if you expect the index
    // value to not exceed Uint8.max.
    static func valueAtIndex(output: CIImage, index: Int) -> UInt8 {
        var bitmap = [UInt8](repeating: 0, count: 4)

        let context = CIContext(options: [CIContextOption.useSoftwareRenderer: true])
        context.render(output, toBitmap: &bitmap, rowBytes: 4, bounds: output.extent, format: CIFormat.RGBA8, colorSpace:  CGColorSpace(name: CGColorSpace.sRGB))
        return bitmap[index]
    }
}
