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
    
    // divideBy divides firstImage by secondImage.
    static func divideBy(firstImage: CIImage?, secondImage: CIImage?) -> CIImage? {
        let comp = CIFilter.divideBlendMode()
        comp.inputImage = secondImage
        comp.backgroundImage = firstImage
        return comp.outputImage
    }
    
    // valueImage converts given sRGB image into Value (in HSV) image.
    static func valueImage(image: CIImage) -> CIImage {
        let gray = CIFilter.maximumComponent()
        gray.inputImage = image
        return gray.outputImage!
    }
    
    // minImage converts given sRGB image into min(RGB) image.
    static func minImage(image: CIImage) -> CIImage {
        let gray = CIFilter.minimumComponent()
        gray.inputImage = image
        return gray.outputImage!
    }
    
    // logicalNOT returns the logical NOT of given image.
    static func logicalNot(image: CIImage?) -> CIImage? {
        let inv = CIFilter.colorInvert()
        inv.inputImage = image
        return inv.outputImage
    }
    
    // histogram calculates historgram of given image extent.
    static func histogram(image: CIImage?) -> CIImage? {
        guard let img = image else {
            return nil
        }
        guard let hist = CIFilter(name: "CIAreaHistogram", parameters: ["inputImage": img, "inputExtent": img.extent, "inputCount": 256]) else {
            print ("Could not create CIAreaHistogram")
            return nil
        }
        return hist.outputImage
    }
    
    // countNonZeroPixels counts number of non-zero pixels in given image.
    static func countNonZeroPixels(image: CIImage?) -> Int {
        guard let hImage = CIImageHelper.histogram(image: image) else {
            return -1
        }
        let val = CIImageHelper.floatValueAtIndex(output: hImage, index: 255)
        if val == -1 {
            return -1
        }
        return Int(val * Float(image!.extent.width) * Float(image!.extent.height))
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
        let val = CIImageHelper.floatValueAtIndex(output: output, index: 0)
        if val == -1 {
            return nil
        }
        return UInt8(val*255.0)
    }
    
    // valueAtIndex returns the Uint8 value at given index of a CIImage that is
    // an output of a filter operation. Use it only if you expect the index
    // value to not exceed Uint8.max.
    static func valueAtIndex(output: CIImage, index: Int) -> UInt8 {
        var bitmap = [UInt8](repeating: 0, count: 4)

        let context = CIContext(options: [CIContextOption.useSoftwareRenderer: true])
        context.render(output, toBitmap: &bitmap, rowBytes: 4, bounds: output.extent, format: CIFormat.RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return bitmap[index]
    }
    
    // floatValueAtIndex returns the float value at given index for given image.
    static func floatValueAtIndex(output: CIImage, index: Int) -> Float {
        // Create temporary pixel buffer and render output to it.
        var temp :CVPixelBuffer? = nil
        let ret = CVPixelBufferCreate(nil, Int(output.extent.width), Int(output.extent.height), kCVPixelFormatType_OneComponent32Float
, nil, &temp)
        if ret != 0 {
            print ("Could not create cvpixelbuffer with return: \(ret)")
            return -1
        }
        guard let buf = temp else {
            print ("CVPixelBuffer returned nil")
            return -1
        }
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        context.render(output, to: buf)
        
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        let base = CVPixelBufferGetBaseAddress(buf)!
        let val = base.assumingMemoryBound(to: Float32.self)[index]
        CVPixelBufferUnlockBaseAddress(buf, .readOnly)
        
        return Float(val)
    }
    
    // averageSaturation returns average Saturation (from HSV) of given sRGB within given mask bounding box.
    static func averageSaturation(rgbImage: CIImage, maskBbox: CGRect) -> Float? {
        let maxImage = CIImageHelper.valueImage(image: rgbImage)
        let minImage = CIImageHelper.minImage(image: rgbImage)
        
        // bitwiseXor for masks can be reused for difference between maxImage and minImage.
        let deltaImage = CIImageHelper.bitwiseXor(firstMask: maxImage, secondMask: minImage)
        guard let satImage = CIImageHelper.divideBy(firstImage: deltaImage, secondImage:  maxImage) else {
            print ("unable to create saturation image")
            return nil
        }
        
        // Find average.
        guard let avg = CIFilter(name: "CIAreaAverage", parameters: ["inputImage": satImage, "inputExtent": CIVector(cgRect: maskBbox)]) else {
            print ("Could not construct CIFilter average")
            return nil
        }
        guard let output = avg.outputImage else {
            print ("averageValue: outputimage nil")
            return nil
        }
        
        return CIImageHelper.floatValueAtIndex(output: output, index: 0)
    }
}
