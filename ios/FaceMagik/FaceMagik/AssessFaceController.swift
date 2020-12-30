//
//  AssessFaceController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/25/20.
//

import UIKit
import Photos
import CoreMotion

protocol FaceProcessorDelegate: AnyObject {
    func detectFace(image: CIImage, depthPixelBuffer: CVPixelBuffer)
    func getNumFaces() -> Int
    func getFaceMask() -> CIImage?
}

class AssessFaceController: UIViewController {
    @IBOutlet var headingLabel: UILabel!
    @IBOutlet var isoLabel: UILabel!
    @IBOutlet var exposureLabel: UILabel!
    
    // Core Motion variables.
    private let motionManager = CMMotionManager()
    private var motionQueue = OperationQueue()
    static private let motionFrequency = 1.0/30.0
    
    // AVCaptureSession variables.
    @IBOutlet weak private var previewView: PreviewMetalView!
    private let captureSessionQueue = DispatchQueue(label: "vision request queue", qos: .userInteractive, attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var captureSession =  AVCaptureSession()
    @objc private var cameraDevice: AVCaptureDevice!
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthDataOutput = AVCaptureDepthDataOutput()
    private let dataOutputQueue = DispatchQueue(label: "synchronized data output queue")
    private var outputSynchronizer: AVCaptureDataOutputSynchronizer!
    private static let FRAME_RATE: Int32 = 20
    
    // Other variables.
    private let notifCenter = NotificationCenter.default
    private var exposureObservation: NSKeyValueObservation?
    private var isoObservation: NSKeyValueObservation?
    var facePropertiesDelegate: FaceProcessorDelegate?
    
    static func storyboardInstance() -> AssessFaceController? {
        let className = String(describing: AssessFaceController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? AssessFaceController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.previewView.rotation = .rotate180Degrees
        
        if !self.motionManager.isDeviceMotionAvailable {
            print ("Device motion unavaible! Error!")
            return
        }
        
        self.notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        self.notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.setupVideoCaptureSession()
        
        self.observeDevice()
        
        self.captureSessionQueue.async {
            self.captureSession.startRunning()
            self.startMotionUpdates()
        }
    }
    
    @objc private func appMovedToBackground() {
        if self.motionManager.isDeviceMotionActive {
            self.motionManager.stopDeviceMotionUpdates()
        }
        self.previewView.image = nil
    }
    
    @objc private func appMovedToForeground() {
        if !self.motionManager.isDeviceMotionActive {
            self.startMotionUpdates()
        }
        if !self.captureSession.isRunning {
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
            self.cameraDevice.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: AssessFaceController.FRAME_RATE)
            self.cameraDevice.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: AssessFaceController.FRAME_RATE)
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
    
    // observeDevice observes the exposure duration and color temperature of current device.
    private func observeDevice() {
        // Start observing camera device exposureDuration.
        self.exposureObservation = observe(\.self.cameraDevice.exposureDuration, options: .new){
            object, change in
            guard let newVal = change.newValue else {
                return
            }
            DispatchQueue.main.async {
                self.exposureLabel.text = "E:" + String(Int(1/(newVal.seconds)))
            }
        }
        
        // Start observing camera device white balance gains.
        self.isoObservation = observe(\.self.cameraDevice.iso, options: .new){
            obj, change in
            guard let newVal = change.newValue else {
                return
            }
            DispatchQueue.main.async {
                self.isoLabel.text = String(Int(newVal))
            }
        }
    }
    
   
    // startMotionUpdates starts to receive motion updates from motion manager.
    private func startMotionUpdates() {
        self.motionManager.deviceMotionUpdateInterval = AssessFaceController.motionFrequency
        self.motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: self.motionQueue, withHandler: { (data, error) in
            guard let validData = data else {
                return
            }
            DispatchQueue.main.async {
                self.headingLabel.text = String(Int(validData.heading))
            }
        })
    }
    
    
    // back allowes user to go back to previous view controller.
    @IBAction func back() {
        self.notifCenter.removeObserver(self)
        self.motionManager.stopDeviceMotionUpdates()
        self.captureSession.stopRunning()
        self.previewView.image = nil
        self.dismiss(animated: true)
    }
}

extension AssessFaceController: AVCaptureDataOutputSynchronizerDelegate {
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        
        guard let delegate = self.facePropertiesDelegate else {
            // No additional processing needed.
            return
        }
        
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
        let rgbImage = CIImage(cvPixelBuffer: videoPixelBuffer)
        
        delegate.detectFace(image: rgbImage, depthPixelBuffer: syncedDepthData.depthData.depthDataMap)
        if delegate.getNumFaces() != 1 {
            // Expected 1 face.
            return
        }
        
        self.previewView.image = delegate.getFaceMask()
    }
}
