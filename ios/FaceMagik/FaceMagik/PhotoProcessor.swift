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
    var faceBoundsMask: CGImage!
    var leftEyeMask: CGImage!
    var rightEyeMask: CGImage!
    var leftEyebrowMask: CGImage!
    var rightEyebrowMask: CGImage!
    var outerLipsMask: CGImage!
    var faceContourMask: CGImage!
    var finalFaceMask: CGImage!
    var overExposedPoints: [CGPoint] = []
    var allFacePoints: [CGPoint] = []
    var overExposedMask: CGImage!
    var faceBoundsRect: CGRect!
    
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
    }
    
    // prepareDetectionRequest creates a face + landmarks detection request.
    func prepareDetectionRequest(_ image: CGImage) {
        self.mainImage = image
        
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
        self.detectionRequests = [faceDetectionRequest]
    }
    
    // detectFace calls Vision to detect face for given image.
    func detectFace() {
        if self.detectionRequests.count == 0 {
            print ("No detection requests found")
            return
        }
        // Create a request handler.
        let imageRequestHandler = VNImageRequestHandler(cgImage: self.mainImage,
                                                        orientation: .up,
                                                        options: [:])
        do {
            try imageRequestHandler.perform(self.detectionRequests)
        } catch let error as NSError {
            print ("Failed to perform Face detection with error: \(error)")
        }
    }
    
    // overExposedPercent returns the percentage of face points that are overexposed.
    // Returns -1 if there was no face detected.
    func overExposedPercent() -> Double {
        if self.allFacePoints.count == 0 {
            return -1.0
        }
        return (Double(self.overExposedPoints.count)/Double(self.allFacePoints.count))*100.0
    }
    
    // createFaceBoundsMask creates a CGImage with given face bounds mask.
    func createFaceBoundsMask(_ normRect: CGRect) -> CGImage? {
        let rect = VNImageRectForNormalizedRect(normRect, Int(self.mainImage.width), Int(self.mainImage.height))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: Int(self.mainImage.width), height: Int(self.mainImage.height)), format: format)

        let img = renderer.image { ctx in
            let rectangle = CGRect(x: CGFloat(Int(rect.minX)), y: CGFloat(self.mainImage.height-Int(rect.maxY)), width: CGFloat(Int(rect.width)), height: CGFloat(Int(rect.height)))
            
            self.faceBoundsRect = rectangle
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
        landmarkPoints = self.toCGCoordinates(landmarkPoints)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: Int(self.mainImage.width), height: Int(self.mainImage.height)), format: format)
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
        faceContourPoints = self.toCGCoordinates(faceContourPoints)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: Int(self.mainImage.width), height: Int(self.mainImage.height)), format: format)
        
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
    
    // createOverExposedMask creates the mask of overexposed points on the face.
    func createOverExposedMask() -> CGImage? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: Int(self.mainImage.width), height: Int(self.mainImage.height)), format: format)
        
        let img = renderer.image { ctx in
            ctx.cgContext.setFillColor(UIColor.white.cgColor)
            for p in self.overExposedPoints {
                let rect = CGRect(x: p.x, y: p.y, width: 1, height: 1)
                ctx.cgContext.addRect(rect)
            }
            ctx.cgContext.drawPath(using: .fill)
        }
        return img.cgImage
    }
    
    // toCGCoordinates maps given points from UIImage coordinate system (left bottom origin)
    // to the CGImage coordinate system (left top origion).
    func toCGCoordinates(_ points: [CGPoint]) -> [CGPoint] {
        points.compactMap({ return CGPoint(x: $0.x, y: CGFloat(CGFloat(self.mainImage.height) - $0.y)) })
    }
    
    // calculateOverExposedPoints calculates overexposed points on the face.
    func calculateOverExposedPoints() {
        guard let mainArr = self.pixelDataArray(cgImage: self.mainImage) else {
            print ("Main image pixel array not found")
            return
        }
        
        guard let faceMaskArr = self.pixelDataArray(cgImage: self.finalFaceMask) else {
            print ("Final face mask pixel array not found")
            return
        }
        
        let div = CGFloat(255.0)
        let numIterations = mainArr.count/4
        let w = self.mainImage.width
        
        var exposedPoints: [CGPoint] = []
        var allPoints: [CGPoint] = []
        for i in 0..<numIterations {
            if (CGFloat(faceMaskArr[4*i])/div < 0.90) && (CGFloat(faceMaskArr[4*i + 1])/div < 0.90) && (CGFloat(faceMaskArr[4*i + 2])/div < 0.90) {
                // Not in face mask, skip.
                continue
            }
            let val = max(mainArr[4*i], mainArr[4*i + 1], mainArr[4*i + 2])
            if val >= 240 {
                exposedPoints.append(CGPoint(x: CGFloat(i % w), y: CGFloat(i/w)))
            }
            allPoints.append(CGPoint(x: CGFloat(i % w), y: CGFloat(i/w)))
        }
        
        self.overExposedPoints = exposedPoints
        self.allFacePoints = allPoints
        
        guard let overExposedMask = self.createOverExposedMask() else {
            print ("Couldnot create over exposed mask")
            return
        }
        self.overExposedMask = overExposedMask
    }
    
    // pixelDataArray returns the image data as an array of pixel values.
    func pixelDataArray(cgImage: CGImage) -> [UInt8]? {
        let dataSize = cgImage.width * cgImage.height * 4
        var pixelData = [UInt8](repeating: 0, count: Int(dataSize))
        guard let context = CGContext(data: &pixelData,
                                width: cgImage.width,
                                height: cgImage.height,
                                bitsPerComponent: 8,
                                bytesPerRow: 4 * cgImage.width,
                                space: CGColorSpace(name: CGColorSpace.displayP3)!,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        return pixelData
    }
    
    // computeFinalFaceMask computes the face mask after combining given portrait mask with all face masks.
    func computeFinalFaceMask(_ portraitMask: CGImage) {
        // blend portrait and face contour mask.
        var comp = CIFilter.minimumCompositing()
        comp.inputImage = CIImage(cgImage: portraitMask)
        comp.backgroundImage = CIImage(cgImage: self.faceContourMask)
        var out = comp.outputImage
        
        // blend face bounds mask.
        comp.inputImage = out
        comp.backgroundImage = CIImage(cgImage: self.faceBoundsMask)
        out = comp.outputImage
        
        // blend left eye mask.
        comp = CIFilter.differenceBlendMode()
        comp.backgroundImage = out
        comp.inputImage = CIImage(cgImage: self.leftEyeMask)
        out = comp.outputImage!
        
        // blend right eye mask.
        comp.backgroundImage = out
        comp.inputImage = CIImage(cgImage: self.rightEyeMask)
        out = comp.outputImage!
        
        // blend left eyebrow mask.
        comp.backgroundImage = out
        comp.inputImage = CIImage(cgImage: self.leftEyebrowMask)
        out = comp.outputImage!
        
        // blend right eyebrow mask.
        comp.backgroundImage = out
        comp.inputImage = CIImage(cgImage: self.rightEyebrowMask)
        out = comp.outputImage!
        
        // blend outer lips mask.
        comp.backgroundImage = out
        comp.inputImage = CIImage(cgImage: self.outerLipsMask)
        out = comp.outputImage!

        if out == nil {
            return
        }
        guard let cgImage = PhotoProcessor.CIImageToCGImage(out!) else {
            return
        }
        self.finalFaceMask = cgImage
    }
    
    // overlayOverExposedMask returns image with over exposed mask overlayed.
    func overlayOverExposedMask() -> UIImage {
        let mainCIImage = CIImage(cgImage: self.mainImage)
        
        let cmap = CIFilter.colorMap()
        cmap.inputImage = mainCIImage
        let makeup = cmap.outputImage
        
        let blend = CIFilter.blendWithMask()
        blend.backgroundImage = mainCIImage
        blend.inputImage = makeup
        blend.maskImage =  CIImage(cgImage: self.overExposedMask)
        
        return UIImage(ciImage: blend.outputImage!)
    }
}
