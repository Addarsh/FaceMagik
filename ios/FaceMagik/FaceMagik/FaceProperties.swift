//
//  FaceProperties.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/28/20.
//

import UIKit
import Vision

class FaceProperties: FaceProcessorDelegate {
    private var image: CIImage?
    private var numFacesFound: Int = 0
    private var faceBoundsMask: CIImage?
    private var leftEyeMask: CIImage?
    private var rightEyeMask: CIImage?
    private var leftEyebrowMask: CIImage?
    private var rightEyebrowMask: CIImage?
    private var outerLipsMask: CIImage?
    private var faceContourMask: CIImage?
    private var finalFaceMask: CIImage?
    private var overExposedMask: CIImage?
    private var faceBoundsRect: CGRect?
    private var depthOfFaceCenter: Float?
    private var depthMaskCIImage: CIImage?
    private var faceMask: CIImage?
    
    // toCGCoordinates maps given points from UIImage coordinate system (left bottom origin)
    // to the CGImage coordinate system (left top origion).
    private static func toCGCoordinates(_ points: [CGPoint], _ imageHeight: CGFloat) -> [CGPoint] {
        points.compactMap({ return CGPoint(x: $0.x, y: imageHeight - $0.y) })
    }
    
    // detectFace calls Vision to detect face for given CIImage.
    func detectFace(image: CIImage, depthPixelBuffer: CVPixelBuffer) {
        self.image = image
        let imageRequestHandler = VNImageRequestHandler(ciImage: image,
                                                        orientation: .up,
                                                        options: [:])
        do {
            try imageRequestHandler.perform([VNDetectFaceLandmarksRequest(completionHandler: { (request, error) in
                
                if error != nil {
                    print("FaceDetection error: \(String(describing: error)).")
                }
                
                guard let faceDetectionRequest = request as? VNDetectFaceLandmarksRequest,
                    let results = faceDetectionRequest.results as? [VNFaceObservation] else {
                        return
                }
                self.numFacesFound = results.count
                if results.count != 1 {
                    print ("Expected 1 face, found \(results.count) faces in image")
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
                
                self.faceBoundsMask = self.createFaceBoundsMask(result.boundingBox)
                self.leftEyeMask = self.createLandmarkMask(leftEye)
                self.rightEyeMask = self.createLandmarkMask(rightEye)
                self.leftEyebrowMask = self.createLandmarkMask(leftEyebrow)
                self.rightEyebrowMask = self.createLandmarkMask(rightEyebrow)
                self.outerLipsMask = self.createLandmarkMask(outerLips)
                self.faceContourMask = self.createFaceContourMask(faceContour)
                self.depthOfFaceCenter = self.computeDepthOfFace(depthPixelBuffer)
                self.depthMaskCIImage = self.createDepthMask(depthPixelBuffer)
            })])
        } catch let error as NSError {
            print ("Failed to perform Face detection with error: \(error)")
        }
    }
    
    // getNumFaces returns the number of faces detected in image.
    func getNumFaces() -> Int {
        return self.numFacesFound
    }
    
    // createFaceBoundsMask creates a CIImage mask with given face bounds rect.
    private func createFaceBoundsMask(_ normRect: CGRect) -> CIImage? {
        guard let width = self.image?.extent.width else {
            return nil
        }
        guard let height = self.image?.extent.height else {
            return nil
        }
        let rect = VNImageRectForNormalizedRect(normRect, Int(width), Int(height))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)

        let img = renderer.image { ctx in
            let rectangle = CGRect(x: rect.minX, y: height-rect.maxY, width: rect.width, height: rect.height)
            
            self.faceBoundsRect = rectangle
            ctx.cgContext.setFillColor(UIColor.white.cgColor)
            ctx.cgContext.addRect(rectangle)
            ctx.cgContext.drawPath(using: .fill)
        }
        guard let cgImage = img.cgImage else {
            print ("facebounds cgimage is nil")
            return nil
        }
        return CIImage(cgImage: cgImage)
    }
    
    // createLandmarkMask returns a CIImage mask of given normalized VNFaceLandmarkRegion2D landmark points.
    private func createLandmarkMask(_ landmark: VNFaceLandmarkRegion2D) -> CIImage? {
        guard let width = self.image?.extent.width else {
            return nil
        }
        guard let height = self.image?.extent.height else {
            return nil
        }
        
        // Convert normalized points to image coordinate space.
        var landmarkPoints = landmark.pointsInImage(imageSize: CGSize(width: width, height: height))
        landmarkPoints = FaceProperties.toCGCoordinates(landmarkPoints, height)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        let img = renderer.image { ctx in
            ctx.cgContext.setFillColor(UIColor.white.cgColor)
            ctx.cgContext.addLines(between: landmarkPoints)
            ctx.cgContext.closePath()
            ctx.cgContext.drawPath(using: .fill)
        }
        guard let cgImage = img.cgImage else {
            print ("landmarks cgimage is nil")
            return nil
        }
        return CIImage(cgImage: cgImage)
    }
    
    // createFaceContourMask creates a CIImage mask of given face contour points.
    private func createFaceContourMask(_ faceContour: VNFaceLandmarkRegion2D) -> CIImage? {
        guard let width = self.image?.extent.width else {
            return nil
        }
        guard let height = self.image?.extent.height else {
            return nil
        }
        
        // Convert normalized points to image coordinate space.
        var faceContourPoints = faceContour.pointsInImage(imageSize: CGSize(width: width, height: height))
        faceContourPoints = FaceProperties.toCGCoordinates(faceContourPoints, height)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        
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
        guard let cgImage = img.cgImage else {
            print ("face contours cgimage is nil")
            return nil
        }
        return CIImage(cgImage: cgImage)
    }
    
    // computeDepthOfFace returns the depth value at the center of the detected face.
    private func computeDepthOfFace(_ depthPixelBuffer: CVPixelBuffer) -> Float? {
        guard let width = self.image?.extent.width else {
            return nil
        }
        guard let faceBoundsRect = self.faceBoundsRect else {
            return nil
        }
        let center = CGPoint(x: faceBoundsRect.midX, y: faceBoundsRect.minY)
        let scale = CGFloat(CVPixelBufferGetWidth(depthPixelBuffer))/width
        let pixelX = Int((center.x * scale).rounded())
        let pixelY = Int((center.y * scale).rounded())
        
        CVPixelBufferLockBaseAddress(depthPixelBuffer, .readOnly)
        
        let rowData = CVPixelBufferGetBaseAddress(depthPixelBuffer)! + pixelY * CVPixelBufferGetBytesPerRow(depthPixelBuffer)
        let faceCenterDepth = rowData.assumingMemoryBound(to: Float32.self)[pixelX]
        CVPixelBufferUnlockBaseAddress(depthPixelBuffer, .readOnly)
        
        return faceCenterDepth
    }
    
    // createDepthMask creates a depth map mask using the face center depth as cutoff (already computed).
    // Every pixel below cutoff is converted to 1. otherwise it's 0.
    private func createDepthMask(_ depthPixelBuffer: CVPixelBuffer) -> CIImage? {
        guard let depthCutOff = self.depthOfFaceCenter else {
            return nil
        }
        guard let width = self.image?.extent.width else {
            return nil
        }
        guard let height = self.image?.extent.height else {
            return nil
        }
        
        let s :CGFloat = -10
        let b = -s*CGFloat(depthCutOff+0.25)
        
        guard let mat = CIFilter(name: "CIColorMatrix", parameters: ["inputImage": CIImage(cvPixelBuffer: depthPixelBuffer), "inputRVector": CIVector(x: s, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: s, z: 0, w: 0),"inputBVector": CIVector(x: 0, y: 0, z: s, w: 0),"inputBiasVector": CIVector(x: b, y: b, z: b, w: 0)]) else {
            print ("Could not construct CIFilter")
            return nil
        }
        let clamp = CIFilter.colorClamp()
        clamp.inputImage = mat.outputImage
        let out = clamp.outputImage
        
        // Scale depth mask to rgb image extent.
        let scale = CGAffineTransform(scaleX: width/CGFloat(CVPixelBufferGetWidth(depthPixelBuffer)), y: height/CGFloat(CVPixelBufferGetHeight(depthPixelBuffer)))
        return out?.transformed(by: scale)
    }
    
    // getFaceMask returns mask of face points excluding eyes, eyebrows, lips and teeth.
    func getFaceMask() -> CIImage? {
        
        // blend depthmask and face contour mask.
        var comp = CIFilter.minimumCompositing()
        comp.inputImage = self.depthMaskCIImage
        comp.backgroundImage = self.faceContourMask
        var out = comp.outputImage
        
        // blend face bounds mask.
        comp.inputImage = out
        comp.backgroundImage = self.faceBoundsMask
        out = comp.outputImage
        
        // blend left eye mask.
        comp = CIFilter.differenceBlendMode()
        comp.backgroundImage = out
        comp.inputImage = self.leftEyeMask
        out = comp.outputImage
        
        // blend right eye mask.
        comp.backgroundImage = out
        comp.inputImage = self.rightEyeMask
        out = comp.outputImage
        
        // blend left eyebrow mask.
        comp.backgroundImage = out
        comp.inputImage = self.leftEyebrowMask
        out = comp.outputImage
        
        // blend right eyebrow mask.
        comp.backgroundImage = out
        comp.inputImage = self.rightEyebrowMask
        out = comp.outputImage
        
        // blend outer lips mask.
        comp.backgroundImage = out
        comp.inputImage = self.outerLipsMask
        out = comp.outputImage
        
        self.faceMask = out
        
        return out
    }
}
