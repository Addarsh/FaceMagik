//
//  PhotoProcessor.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 10/30/20.
//

import Foundation
import Vision
import UIKit
import CoreImage.CIFilterBuiltins

class PhotoProcessor: NSObject {
    var semaphore = DispatchSemaphore(value: 0)
    var detectionRequests: [VNDetectFaceLandmarksRequest] = []
    var numFaces = 0
    var mainImage: CGImage!
    var portraitMask: CGImage!
    var faceBoundsMask: CGImage!
    var leftEyeMask: CGImage!
    var rightEyeMask: CGImage!
    var leftEyebrowMask: CGImage!
    var rightEyebrowMask: CGImage!
    var outerLipsMask: CGImage!
    var faceContourMask: CGImage!
    
    // CIImageToCGImage converts CIImage to CGImage.
    static func CIImageToCGImage(_ image: CIImage) -> CGImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(image, from: image.extent) {
            return cgImage
        }
        return nil
    }
    
    // CGImageToUIImage converts CGImage to UIImage.
    static func CGImageToUIImage(_ image: CGImage) -> UIImage? {
        let ciImage = CIImage(cgImage: image)
        let context = CIContext(options: [CIContextOption.useSoftwareRenderer: true])
        
        guard let pngData = context.pngRepresentation(of: ciImage, format: CIFormat.ARGB8, colorSpace: CGColorSpace(name: CGColorSpace.displayP3)!) else {
            return nil
        }
        return UIImage(data: pngData)
    }
    
    override init() {
        super.init()
        
        prepareDetectionRequest()
    }
    
    func prepareDetectionRequest() {
        let faceDetectionRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request, error) in
            defer { self.semaphore.signal() }
            
            if error != nil {
                print("FaceDetection error: \(String(describing: error)).")
            }
            
            guard let faceDetectionRequest = request as? VNDetectFaceLandmarksRequest,
                let results = faceDetectionRequest.results as? [VNFaceObservation] else {
                    return
            }
            self.numFaces = results.count
            if results.count != 1 {
                print ("Error! Want 1 Face in image, Got \(results.count) Faces in image!")
                return
            }
            let result = results[0]
            guard let landmarks = result.landmarks else {
                return
            }
            guard let leftEye = landmarks.leftEye else {
                print ("Left Eye not found in image")
                return
            }
            guard let rightEye = landmarks.rightEye else {
                print ("Right Eye not found in image")
                return
            }
            guard let leftEyebrow = landmarks.leftEyebrow else {
                print ("Left Eyebrow not found in image")
                return
            }
            guard let rightEyebrow = landmarks.rightEyebrow else {
                print ("Right Eyebrow not found in image")
                return
            }
            guard let outerLips = landmarks.outerLips else {
                print ("outerLips not found in image")
                return
            }
            guard let faceContour = landmarks.faceContour else {
                print ("faceContour not found in image")
                return
            }
            
            guard let faceBoundsMask = self.createFaceBoundsMask(result.boundingBox) else {
                print ("Could not create face bounds mask")
                return
            }
            self.faceBoundsMask = faceBoundsMask
            
            guard let leftEyeMask = self.createLandmarkMask(leftEye) else {
                print ("Could not create left eye mask")
                return
            }
            self.leftEyeMask = leftEyeMask
            
            guard let rightEyeMask = self.createLandmarkMask(rightEye) else {
                print ("Could not create right eye mask")
                return
            }
            self.rightEyeMask = rightEyeMask
            
            guard let leftEyebrowMask = self.createLandmarkMask(leftEyebrow) else {
                print ("Could not create left eyebrow mask")
                return
            }
            self.leftEyebrowMask = leftEyebrowMask
            
            guard let rightEyebrowMask = self.createLandmarkMask(rightEyebrow) else {
                print ("Could not create right eyebrow mask")
                return
            }
            self.rightEyebrowMask = rightEyebrowMask
            
            guard let outerLipsMask = self.createLandmarkMask(outerLips) else {
                print ("Could not create outer lips mask")
                return
            }
            self.outerLipsMask = outerLipsMask
            
            guard let faceContourMask = self.createFaceContourMask(faceContour) else {
                print ("Could not create face contour mask")
                return
            }
            self.faceContourMask = faceContourMask
            
        })
        detectionRequests = [faceDetectionRequest]
    }
    
    func detectFace(_ image: CGImage, _ mask: CGImage) {
        if detectionRequests.count == 0 {
            print ("No detection requests found")
            return
        }
        mainImage = image
        portraitMask = mask
        // Create a request handler.
        let imageRequestHandler = VNImageRequestHandler(cgImage: image,
                                                        orientation: .up,
                                                        options: [:])
        do {
            try imageRequestHandler.perform(detectionRequests)
        } catch let error as NSError {
            print ("Failed to perform Face detection with error: \(error)")
        }
    }
    
    // createFaceBoundsMask creates a CGImage with given face bounds mask.
    func createFaceBoundsMask(_ normRect: CGRect) -> CGImage? {
        let rect = VNImageRectForNormalizedRect(normRect, Int(mainImage.width), Int(mainImage.height))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: Int(mainImage.width), height: Int(mainImage.height)), format: format)

        let img = renderer.image { ctx in
            let rectangle = CGRect(x: CGFloat(Int(rect.minX)), y: CGFloat(mainImage.height-Int(rect.maxY)), width: CGFloat(Int(rect.width)), height: CGFloat(Int(rect.height)))
            ctx.cgContext.setFillColor(UIColor.white.cgColor)
            ctx.cgContext.addRect(rectangle)
            ctx.cgContext.drawPath(using: .fill)
        }
        return img.cgImage
    }
    
    // createLandmarkMask returns a CGImage mask of given normalized VNFaceLandmarkRegion2D landmark points.
    func createLandmarkMask(_ landmark: VNFaceLandmarkRegion2D) -> CGImage? {
        // Convert normalized points to image coordinate space.
        var landmarkPoints = landmark.pointsInImage(imageSize: CGSize(width: CGFloat(self.mainImage.width), height: CGFloat(self.mainImage.height)))
        landmarkPoints = toCGCoordinates(landmarkPoints)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: Int(mainImage.width), height: Int(mainImage.height)), format: format)
        let img = renderer.image { ctx in
            ctx.cgContext.setFillColor(UIColor.white.cgColor)
            
            ctx.cgContext.addLines(between: landmarkPoints)
            ctx.cgContext.closePath()
            ctx.cgContext.drawPath(using: .fill)
        }
        return img.cgImage
    }
    
    // createFaceBoundsMask creates a CGImage with given face contours mask.
    func createFaceContourMask(_ faceContour: VNFaceLandmarkRegion2D) -> CGImage? {
        // Convert normalized points to image coordinate space.
        var faceContourPoints = faceContour.pointsInImage(imageSize: CGSize(width: CGFloat(self.mainImage.width), height: CGFloat(self.mainImage.height)))
        faceContourPoints = toCGCoordinates(faceContourPoints)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: Int(mainImage.width), height: Int(mainImage.height)), format: format)
        
        let img = renderer.image { ctx in
            ctx.cgContext.setFillColor(UIColor.white.cgColor)
            
            ctx.cgContext.addLines(between: faceContourPoints)
            ctx.cgContext.closePath()
            ctx.cgContext.drawPath(using: .fill)
            
            if faceContourPoints.count < 2 {
                print ("Want >= 2 face contour points, got \(faceContourPoints.count) points")
                return
            }
            
            // Draw extra rectangle on top of face contour as mask.
            let leftcheekPoint = faceContourPoints.first!
            let rightCheekPoint = faceContourPoints.last!
            let rect = CGRect(x: leftcheekPoint.x, y: 0, width: rightCheekPoint.x-leftcheekPoint.x, height: max(leftcheekPoint.y, rightCheekPoint.y))
            ctx.cgContext.addRect(rect)
            ctx.cgContext.drawPath(using: .fill)
        }
        return img.cgImage
    }
    
    // toCGCoordinates maps given points from UIImage coordinate system (left bottom origin)
    // to the CGImage coordinate system (left top origion).
    func toCGCoordinates(_ points: [CGPoint]) -> [CGPoint] {
        points.compactMap({ return CGPoint(x: $0.x, y: CGFloat(CGFloat(mainImage.height) - $0.y)) })
    }
    
    // overExposurePercent returns the percentage amount by which given main image is overexposed.
    func overExposurePercent() -> Double {
        guard let providerData = mainImage.dataProvider?.data
        else {
            print ("mainImage data provider not found")
            return -1.0
        }
        guard let data = CFDataGetBytePtr(providerData) else {
            print ("CGData Pointer not found for main image providerData")
            return -1.0
        }
        guard let maskProviderData = portraitMask.dataProvider?.data else {
            print ("portrait mask data provider not found")
            return -1.0
        }
        guard let maskData = CFDataGetBytePtr(maskProviderData) else {
            print ("CGData Pointer not found for portrait mask providerData")
            return -1.0
        }
        
        let numComponents = 4
        let div = CGFloat(255.0)
        let w = mainImage.width
        let h = mainImage.height
        
        var totalPoints = 0
        var brighterPoints = 0
        for i in 0..<w {
            for j in 0..<h {
                let position = ((w*j) + i)*numComponents
                if CGFloat(maskData[position])/div < 0.99 || CGFloat(maskData[position + 1])/div < 0.99 && CGFloat(maskData[position + 2])/div < 0.99 {
                    // Outside portrait, skip.
                    continue
                }
                let val = max(CGFloat(data[position]), CGFloat(data[position+1]), CGFloat(data[position+2]))
                if val >= 240 {
                    brighterPoints += 1
                }
                totalPoints += 1
            }
        }
        return (Double(brighterPoints)/Double(totalPoints))*100.0
    }
    
    func blendMask() -> UIImage {
        semaphore.wait()
        
        let mainCIImage = CIImage(cgImage: mainImage)
        
        let cmap = CIFilter.colorMap()
        cmap.inputImage = mainCIImage
        let makeup = cmap.outputImage
        
        // portrait mask.
        var blend = CIFilter.blendWithMask()
        blend.backgroundImage = mainCIImage
        blend.inputImage = makeup
        blend.maskImage = CIImage(cgImage: portraitMask)
        var res = blend.outputImage!
        
        // face bounds mask.
        blend = CIFilter.blendWithMask()
        blend.backgroundImage = mainCIImage
        blend.inputImage = res
        blend.maskImage = CIImage(cgImage: faceBoundsMask)
        res = blend.outputImage!
        
        // left eye mask.
        blend = CIFilter.blendWithMask()
        blend.backgroundImage = res
        blend.inputImage = mainCIImage
        blend.maskImage = CIImage(cgImage: leftEyeMask)
        res = blend.outputImage!
        
        // right eye mask.
        blend = CIFilter.blendWithMask()
        blend.backgroundImage = res
        blend.inputImage = mainCIImage
        blend.maskImage = CIImage(cgImage: rightEyeMask)
        res = blend.outputImage!
        
        // left eye brow mask.
        blend = CIFilter.blendWithMask()
        blend.backgroundImage = res
        blend.inputImage = mainCIImage
        blend.maskImage = CIImage(cgImage: leftEyebrowMask)
        res = blend.outputImage!
        
        // right eye brow mask.
        blend = CIFilter.blendWithMask()
        blend.backgroundImage = res
        blend.inputImage = mainCIImage
        blend.maskImage = CIImage(cgImage: rightEyebrowMask)
        res = blend.outputImage!
        
        // outer lips mask.
        blend = CIFilter.blendWithMask()
        blend.backgroundImage = res
        blend.inputImage = mainCIImage
        blend.maskImage = CIImage(cgImage: outerLipsMask)
        res = blend.outputImage!
        
        // face contour mask.
        blend = CIFilter.blendWithMask()
        blend.backgroundImage = mainCIImage
        blend.inputImage = res
        blend.maskImage = CIImage(cgImage: faceContourMask)
        res = blend.outputImage!
        
        return UIImage(ciImage: res)
    }
}
