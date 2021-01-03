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
    func getFaceCenterDepth() -> Float?
}

class AssessFaceController: UIViewController {
    @IBOutlet var headingLabel: UILabel!
    @IBOutlet var isoLabel: UILabel!
    @IBOutlet var tempLabel: UILabel!
    @IBOutlet var exposureLabel: UILabel!
    @IBOutlet private var progressView: UIProgressView!
    
    // Core Motion variables.
    private let motionManager = CMMotionManager()
    private var motionQueue = OperationQueue()
    static private let motionFrequency = 1.0/30.0
    static private let totalHeadingVals = 320
    
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
    
    // Env variables.
    private let notifCenter = NotificationCenter.default
    private var exposureObservation: NSKeyValueObservation?
    private var isoObservation: NSKeyValueObservation?
    private var tempObservation: NSKeyValueObservation?
    private var sensorMap: [Int: SensorValues] = [:]
    private var currTemp: Int = 0
    private var currISO: Int = 0
    private var currExposure: Int = 0
    private let envQueue = DispatchQueue(label: "Env Sensor Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    static private let expPercentThreshold = 70
    static private let isoPerentThreshold = 70
    static private let colorTempThreshold = 70
    
    var facePropertiesDelegate: FaceProcessorDelegate?
    
    
    // Phone too close related variables.
    private var phoneTooCloseAlert: AlertViewController?
    
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
            self.envQueue.async {
                self.currExposure = Int(1/(newVal.seconds))
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
            self.envQueue.async {
                self.currISO = Int(newVal)
            }
            DispatchQueue.main.async {
                self.isoLabel.text = "ISO:" + String(Int(newVal))
            }
        }
        
        // Start observing camera device white balance gains.
        self.tempObservation = observe(\.self.cameraDevice.deviceWhiteBalanceGains, options: .new){
            obj, chng in
            let temp = self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature
            self.envQueue.async {
                self.currTemp = Int(temp)
            }
            DispatchQueue.main.async {
                self.tempLabel.text = String(Int(temp)) + "K"
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
            let heading = Int(validData.heading)
            self.envQueue.async {
                if self.sensorMap.keys.count >= AssessFaceController.totalHeadingVals {
                    // Completed sensor data collection.
                    return
                }
                if self.sensorMap[heading] != nil {
                    return
                }
                self.sensorMap[heading] = SensorValues(iso: self.currISO, exposure: self.currExposure, temp: self.currTemp, sceneType: SceneType.Unknown)
                
                let kCount = self.sensorMap.keys.count
                DispatchQueue.main.async {
                    self.progressView.setProgress(Float(kCount)/Float(AssessFaceController.totalHeadingVals), animated: true)
                }
                if kCount == AssessFaceController.totalHeadingVals {
                    self.validateEnv()
                }
            }
        })
    }
    
    // validateEnv validates if environment is suitable for pictures using existing sensor values.
    // Returns true/false values for if env is indoors, in daylight and well lit respectively.
    private func validateEnv() {
        var numVisibleColorTemp = 0
        var numGoodISO = 0
        var numGoodExposure = 0
        for (_, readouts) in self.sensorMap {
            if readouts.temp >= 4000 {
                numVisibleColorTemp += 1
            }
            if readouts.iso < 400 {
                numGoodISO += 1
            }
            if readouts.exposure <= 50 {
                numGoodExposure += 1
            }
        }

        let colorTempPercent = Int((Float(numVisibleColorTemp)/Float(self.sensorMap.keys.count))*100.0)
        let isoPercent = Int((Float(numGoodISO)/Float(self.sensorMap.keys.count))*100.0)
        let expPercent = Int((Float(numGoodExposure)/Float(self.sensorMap.keys.count))*100.0)
        
        DispatchQueue.main.async {
            self.displayResults(isIndoors: true, isDayLight: colorTempPercent >= AssessFaceController.colorTempThreshold ? true : false, isGoodISO: isoPercent >= AssessFaceController.isoPerentThreshold ? true : false, isGoodExposure: expPercent >= AssessFaceController.expPercentThreshold ? true : false)
        }
    }
    
    // displayResults displays results of light testing.
    private func displayResults(isIndoors: Bool, isDayLight: Bool, isGoodISO: Bool, isGoodExposure: Bool) {
        guard let vc = LightingResultsController.storyboardInstance() else {
            return
        }
        vc.isIndoors = isIndoors
        vc.isDayLight = isDayLight
        vc.isGoodISO = isGoodISO
        vc.isGoodExposure = isGoodExposure
        vc.modalPresentationStyle = .fullScreen
        self.present(vc, animated: true)
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
        guard let faceDepth = delegate.getFaceCenterDepth() else {
            // Face depth value not found.
            return
        }
        self.previewView.image = rgbImage
        
        if isPhoneTooClose(faceDepth: faceDepth) {
            // Wait for user to move phone further away.
            return
        }
    }
    
    // isPhoneTooClose checks if phone is too close to the user and if so, alerts the user.
    // If not, it dismisses any existing alerts.
    private func isPhoneTooClose(faceDepth: Float) -> Bool {
        if faceDepth < 0.25 {
            // phone is too close.
            DispatchQueue.main.async {
                if self.phoneTooCloseAlert != nil {
                    // Alert controller already presented.
                    return
                }
                guard let vc = AlertViewController.storyboardInstance() else {
                    return
                }
                //vc.modalPresentationStyle = .fullScreen
                self.phoneTooCloseAlert = vc
                self.present(vc, animated: true)
            }
            return true
        }
        DispatchQueue.main.async {
            if self.phoneTooCloseAlert == nil {
                // Alert controller already dismissed/
                return
            }
            self.phoneTooCloseAlert?.dismiss(animated: true, completion: nil)
            self.phoneTooCloseAlert = nil
        }
        return false
    }
}
