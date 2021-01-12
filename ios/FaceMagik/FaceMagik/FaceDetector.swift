//
//  FaceDetector.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/28/20.
//

import UIKit
import Vision
import Photos

// FaceProperties stores easy to access properties of a face.
struct FaceProperties {
    var image: CIImage
    var faceDepth: Float
    var fullFaceMask: CIImage
    var leftCheekMask: CIImage
    var rightCheekMask: CIImage
}

class FaceDetector: NSObject, FaceProcessor {
    // AVCaptureSession variables.
    private let captureSessionQueue = DispatchQueue(label: "vision request queue", qos: .userInteractive, attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var captureSession =  AVCaptureSession()
    @objc private var cameraDevice: AVCaptureDevice!
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthDataOutput = AVCaptureDepthDataOutput()
    private let dataOutputQueue = DispatchQueue(label: "synchronized data output queue")
    private var outputSynchronizer: AVCaptureDataOutputSynchronizer!
    private static let FRAME_RATE: Int32 = 20
    private var delegate: FaceProcessorDelegate?
    private var firstFrame: Bool = true
    
    // Landmark regions.
    private var noseLandmarks: VNFaceLandmarkRegion2D!
    private var rightEyeLandmarks: VNFaceLandmarkRegion2D!
    private var leftEyeLandmarks: VNFaceLandmarkRegion2D!
    private var faceContourLandmarks: VNFaceLandmarkRegion2D!
    
    // Face properties.
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
    private var fullFaceMask: CIImage?
    private var leftCheekMask: CIImage?
    private var rightCheekMask: CIImage?
    
    // toCGCoordinates maps given face landmark to the CGImage coordinate system (left top origion).
    private func toCGCoordinates(_ landmark: VNFaceLandmarkRegion2D) -> [CGPoint] {
        guard let width = self.image?.extent.width else {
            return []
        }
        guard let height = self.image?.extent.height else {
            return []
        }
        
        let points = landmark.pointsInImage(imageSize: CGSize(width: width, height: height))
        return points.compactMap({ return CGPoint(x: $0.x, y: height - $0.y) })
    }
    
    // bbox returns a bounding box of given face landmark.
    private func bbox(landmark: VNFaceLandmarkRegion2D) -> CGRect? {
        // Convert normalized points to image coordinate space.
        let landmarkPoints = self.toCGCoordinates(landmark)
        
        guard let xMin = landmarkPoints.min(by: {p1, p2 in p1.x < p2.x})?.x else {
            return nil
        }
        guard let xMax = landmarkPoints.max(by: {p1, p2 in p1.x < p2.x})?.x else {
            return nil
        }
        guard let yMin = landmarkPoints.min(by: {p1, p2 in p1.y < p2.y})?.y else {
            return nil
        }
        guard let yMax = landmarkPoints.max(by: {p1, p2 in p1.y < p2.y})?.y else {
            return nil
        }
        return CGRect(x: xMin, y: yMin, width: xMax - xMin, height: yMax - yMin)
    }
    
    func startDetection(vc: FaceProcessorDelegate?) {
        self.delegate = vc
        
        self.setupVideoCaptureSession()
        
        self.captureSessionQueue.async {
            self.captureSession.startRunning()
        }
    }
    
    // setupVideoCaptureSession sets up a capture session to capture video.
    private func setupVideoCaptureSession() {
        self.captureSession.beginConfiguration()
        
        // Add capture session input.
        guard let dev = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            return
        }
        self.cameraDevice = dev
        
        // Add capture session input.
        guard let captureInput = try? AVCaptureDeviceInput(device: self.cameraDevice), self.captureSession.canAddInput(captureInput) else {
            return
        }
        self.captureSession.addInput(captureInput)
        
        // Add capture session output.
        self.videoOutput.alwaysDiscardsLateVideoFrames = true
        guard self.captureSession.canAddOutput(self.videoOutput) else {
            return
        }
        
        // Set sRGB as default color space.
        self.captureSession.automaticallyConfiguresCaptureDeviceForWideColor = false
        self.captureSession.sessionPreset = .hd1280x720
        self.captureSession.addOutput(self.videoOutput)
            
        if let videoConnection = self.videoOutput.connection(with: .video) {
            videoConnection.videoOrientation = .portrait
            videoConnection.isEnabled = true
        }
        
        // Set sRGB as default color space.
        do {
            try self.cameraDevice.lockForConfiguration()
            self.cameraDevice.activeColorSpace = .sRGB
            self.cameraDevice.unlockForConfiguration()
        } catch {
            print("Error! Could not lock device for configuration: \(error)")
            return
        }
        
        // Add depth data output.
        //self.depthDataOutput.alwaysDiscardsLateDepthData = true
        self.depthDataOutput.isFilteringEnabled = true
        self.captureSession.addOutput(self.depthDataOutput)
        
        if let depthConnection = self.depthDataOutput.connection(with: .depthData) {
            depthConnection.videoOrientation = .portrait
            depthConnection.isEnabled = true
        }
        
        // Search for highest resolution with floating-point depth values
        let depthFormats = self.cameraDevice.activeFormat.supportedDepthDataFormats
        let depth32formats = depthFormats.filter({
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32
        })
        if depth32formats.isEmpty {
            print("Error! Device does not support Float32 depth format")
            return
        }
        
        let selectedFormat = depth32formats.max(by: { first, second in
            CMVideoFormatDescriptionGetDimensions(first.formatDescription).width <
                CMVideoFormatDescriptionGetDimensions(second.formatDescription).width })
        
        do {
            try self.cameraDevice.lockForConfiguration()
            self.cameraDevice.activeDepthDataFormat = selectedFormat
            self.cameraDevice.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: FaceDetector.FRAME_RATE)
            self.cameraDevice.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: FaceDetector.FRAME_RATE)
            self.cameraDevice.unlockForConfiguration()
        } catch {
            print("Error! Could not lock device for configuration: \(error)")
            return
        }
        
        // Use an AVCaptureDataOutputSynchronizer to synchronize the video data and depth data outputs.
        // The first output in the dataOutputs array, in this case the AVCaptureVideoDataOutput, is the "master" output.
        self.outputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [self.videoOutput, self.depthDataOutput])
        self.outputSynchronizer.setDelegate(self, queue: self.dataOutputQueue)
        
        self.captureSession.commitConfiguration()
    }
    
    // detectFace calls Vision to detect face.
    private func detectFace(depthPixelBuffer: CVPixelBuffer) {
        guard let image = self.image else {
            return
        }
        
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
                self.leftEyeLandmarks = leftEye
                
                guard let rightEye = landmarks.rightEye else {
                    print ("Right Eye not found in image")
                    return
                }
                self.rightEyeLandmarks = rightEye
                
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
                guard let nose = landmarks.nose else {
                    print ("nose not found in image")
                    return
                }
                self.noseLandmarks = nose
                
                guard let faceContour = landmarks.faceContour else {
                    print ("faceContour not found in image")
                    return
                }
                self.faceContourLandmarks = faceContour
                
                self.faceBoundsMask = self.createFaceBoundsMask(result.boundingBox)
                self.leftEyeMask = self.createLandmarkMask(leftEye)
                self.rightEyeMask = self.createLandmarkMask(rightEye)
                self.leftEyebrowMask = self.createLandmarkMask(leftEyebrow)
                self.rightEyebrowMask = self.createLandmarkMask(rightEyebrow)
                self.outerLipsMask = self.createLandmarkMask(outerLips)
                self.faceContourMask = self.createFaceContourMask(faceContour)
                self.depthOfFaceCenter = self.computeDepthOfFace(depthPixelBuffer)
                self.depthMaskCIImage = self.createDepthMask(depthPixelBuffer)
                self.fullFaceMask = self.getFaceMask()
                self.leftCheekMask = self.getLeftCheekMask()
                self.rightCheekMask = self.getRightCheekMask()
            })])
        } catch let error as NSError {
            print ("Failed to perform Face detection with error: \(error)")
        }
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
        let landmarkPoints = self.toCGCoordinates(landmark)
        
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
        let faceContourPoints = self.toCGCoordinates(faceContour)
        
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
    
    // getLeftCheekMask returns a CIImage of left cheek.
    private func getLeftCheekMask() -> CIImage? {
        guard let width = self.image?.extent.width else {
            return nil
        }
        guard let height = self.image?.extent.height else {
            return nil
        }
        
        // Due to some weird lateral inversion/flipped coordinate system along x axis, righteyelandmarks
        // appear on the left side. TODO: Dig into this later.
        // X-axis from right to left, Y-axis from top to bottom.
        guard let leftEyeBbox = self.bbox(landmark: self.rightEyeLandmarks) else {
            return nil
        }
        guard let noseBbox = self.bbox(landmark: self.noseLandmarks) else {
            return nil
        }
        let faceContourPoints = self.toCGCoordinates(self.faceContourLandmarks)
        
        // Create left cheek CGRect image.
        let xMin = max(leftEyeBbox.minX, noseBbox.maxX)
        let xMax = faceContourPoints.first!.x
        let yMin = leftEyeBbox.minY + leftEyeBbox.height + 20
        let yMax = noseBbox.minY + noseBbox.height
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        let img = renderer.image { ctx in
            ctx.cgContext.setFillColor(UIColor.white.cgColor)
            ctx.cgContext.addRect(CGRect(x: xMin, y: yMin, width: xMax-xMin, height: yMax-yMin))
            ctx.cgContext.drawPath(using: .fill)
        }
        guard let cgImage = img.cgImage else {
            print ("landmarks cgimage is nil")
            return nil
        }
        
        // Bitwise AND with fullfaceMask to ensure all points in mask lie inside face.
        let comp = CIFilter.minimumCompositing()
        comp.backgroundImage = self.fullFaceMask
        comp.inputImage = CIImage(cgImage: cgImage)
        return comp.outputImage
    }
    
    // getRightCheekMask returns a CIImage of right cheek.
    private func getRightCheekMask() -> CIImage? {
        guard let width = self.image?.extent.width else {
            return nil
        }
        guard let height = self.image?.extent.height else {
            return nil
        }
        
        // Due to some weird lateral inversion/flipped coordinate system along x axis, leftEyelandmarks
        // appear on the right side. TODO: Dig into this later.
        // X-axis from right to left, Y-axis from top to bottom.
        guard let rightEyeBbox = self.bbox(landmark: self.leftEyeLandmarks) else {
            return nil
        }
        guard let noseBbox = self.bbox(landmark: self.noseLandmarks) else {
            return nil
        }
        let faceContourPoints = self.toCGCoordinates(self.faceContourLandmarks)
        
        // Create left cheek CGRect image.
        let xMin = faceContourPoints.last!.x
        let xMax = min(rightEyeBbox.maxX, noseBbox.minX)
        let yMin = rightEyeBbox.minY + rightEyeBbox.height + 20
        let yMax = noseBbox.minY + noseBbox.height
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        let img = renderer.image { ctx in
            ctx.cgContext.setFillColor(UIColor.white.cgColor)
            ctx.cgContext.addRect(CGRect(x: xMin, y: yMin, width: xMax-xMin, height: yMax-yMin))
            ctx.cgContext.drawPath(using: .fill)
        }
        guard let cgImage = img.cgImage else {
            print ("landmarks cgimage is nil")
            return nil
        }
        
        // Bitwise AND with fullfaceMask to ensure all points in mask lie inside face.
        let comp = CIFilter.minimumCompositing()
        comp.backgroundImage = self.fullFaceMask
        comp.inputImage = CIImage(cgImage: cgImage)
        return comp.outputImage
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
    private func getFaceMask() -> CIImage? {
        
        // blend depthmask and face contour mask.
        var out = CIImageHelper.bitwiseAnd(firstMask: self.faceContourMask, secondMask: self.depthMaskCIImage)
        
        // blend face bounds mask.
        out = CIImageHelper.bitwiseAnd(firstMask: out, secondMask: self.faceBoundsMask)
        
        // blend left eye mask.
        out = CIImageHelper.bitwiseXor(firstMask: out, secondMask: self.leftEyeMask)
        
        // blend right eye mask.
        out = CIImageHelper.bitwiseXor(firstMask: out, secondMask: self.rightEyeMask)
        
        // blend left eyebrow mask.
        out = CIImageHelper.bitwiseXor(firstMask: out, secondMask: self.leftEyebrowMask)
        
        // blend right eyebrow mask.
        out = CIImageHelper.bitwiseXor(firstMask: out, secondMask: self.rightEyebrowMask)
        
        // blend outer lips mask.
        out = CIImageHelper.bitwiseXor(firstMask: out, secondMask: self.outerLipsMask)
        
        return out
    }
    
    // getDevice returns camera device instance.
    func getDevice() -> AVCaptureDevice {
        return self.cameraDevice
    }
    
    func stop() {
        self.captureSession.stopRunning()
        self.image = nil
        self.firstFrame = true
    }
    
    func resume() {
        if !self.captureSession.isRunning {
            self.captureSession.startRunning()
        }
    }
}

extension FaceDetector: AVCaptureDataOutputSynchronizerDelegate {
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        
        // Check video frame.
        guard let syncedVideoData = synchronizedDataCollection.synchronizedData(for: self.videoOutput) as? AVCaptureSynchronizedSampleBufferData else {
            return
        }
        // Check depth data frame.
        guard let syncedDepthData = synchronizedDataCollection.synchronizedData(for: self.depthDataOutput) as? AVCaptureSynchronizedDepthData else {
            return
        }
        if syncedVideoData.sampleBufferWasDropped || syncedDepthData.depthDataWasDropped {
            return
        }
        guard let videoPixelBuffer = CMSampleBufferGetImageBuffer(syncedVideoData.sampleBuffer) else {
            print ("Could not convert video sample buffer to cvpixelbuffer")
            return
        }
        let rgbImage = CIImage(cvPixelBuffer: videoPixelBuffer.copy())
        self.image = rgbImage
        
        self.detectFace(depthPixelBuffer: syncedDepthData.depthData.depthDataMap.copy())
        if self.numFacesFound != 1 {
            // Expected 1 face.
            return
        }
        
        if self.firstFrame {
            self.delegate?.firstFrame()
            self.firstFrame = false
        }
        self.image = rgbImage
        
        guard let faceProperties = self.constructFaceProperties() else {
            print ("failed to construct face properties")
            return
        }
        
        self.delegate?.frameUpdated(faceProperties: faceProperties)
    }
    
    private func constructFaceProperties() -> FaceProperties? {
        guard let image = self.image else {
            print ("rgbimage not found")
            return nil
        }
        guard let faceDepth = self.depthOfFaceCenter else {
            print ("face depth value not found")
            return nil
        }
        guard let fullFaceMask = self.fullFaceMask else {
            print ("full face mask missing")
            return nil
        }
        guard let leftCheekMask = self.leftCheekMask else {
            print ("left cheek mask missing")
            return nil
        }
        guard let rightCheekMask = self.rightCheekMask else {
            print ("right cheek mask missing")
            return nil
        }
        return FaceProperties(image: image, faceDepth: faceDepth, fullFaceMask: fullFaceMask, leftCheekMask: leftCheekMask, rightCheekMask: rightCheekMask)
    }
}

extension CVPixelBuffer {
    func copy() -> CVPixelBuffer {
        precondition(CFGetTypeID(self) == CVPixelBufferGetTypeID(), "copy() cannot be called on a non-CVPixelBuffer")

        var _copy : CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            CVPixelBufferGetWidth(self),
            CVPixelBufferGetHeight(self),
            CVPixelBufferGetPixelFormatType(self),
            nil,
            &_copy)

        guard let copy = _copy else { fatalError() }

        CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags.readOnly)
        CVPixelBufferLockBaseAddress(copy, CVPixelBufferLockFlags(rawValue: 0))


        let copyBaseAddress = CVPixelBufferGetBaseAddress(copy)
        let currBaseAddress = CVPixelBufferGetBaseAddress(self)

        memcpy(copyBaseAddress, currBaseAddress, CVPixelBufferGetDataSize(copy))

        CVPixelBufferUnlockBaseAddress(copy, CVPixelBufferLockFlags(rawValue: 0))
        CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags.readOnly)


        return copy
    }
}
