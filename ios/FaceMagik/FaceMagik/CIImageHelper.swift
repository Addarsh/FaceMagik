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
}
