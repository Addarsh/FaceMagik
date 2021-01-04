//
//  AssessFaceController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/25/20.
//

import UIKit
import Photos

protocol FaceProcessorDelegate: AnyObject {
    func detectFace(image: CIImage, depthPixelBuffer: CVPixelBuffer)
    func getNumFaces() -> Int
    func getFaceMask() -> CIImage?
    func getFaceCenterDepth() -> Float?
}

protocol EnvObserver {
    func observeLighting(device: AVCaptureDevice, vc: EnvObserverDelegate?)
    func startMotionUpdates()
    func stopMotionUpdates()
}

protocol EnvObserverDelegate {
    func notifyISOUpdate(newISO: Int)
    func notifyExposureUpdate(newExpsosure: Int)
    func notifyTempUpdate(newTemp: Int)
    func notifyProgress(progress: Float)
    func notifyLightingTestResults(isIndoors: Bool, isDayLight: Bool, isGoodISO: Bool, isGoodExposure: Bool)
    func daylightUpdated(isDaylight: Bool)
}

class AssessFaceController: UIViewController {
    enum State {
        case Unknown
        case StartTurnAround
        case IsInDaylight
        case NotInDaylight
    }
    
    @IBOutlet private var isoLabel: UILabel!
    @IBOutlet private var tempLabel: UILabel!
    @IBOutlet private var exposureLabel: UILabel!
    @IBOutlet private var progressView: UIProgressView!
    @IBOutlet private var instructions: UILabel!
    
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
    
    private let notifCenter = NotificationCenter.default
    var facePropertiesDelegate: FaceProcessorDelegate?
    var envObserver: EnvObserver?
    private var phoneTooCloseAlert: AlertViewController?
    
    private var state: State = .Unknown
    private let stateQueue = DispatchQueue(label: "State Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private let unknownPrompt = "Waiting to detect face"
    private let turnAroundPrompt = "Turn Around 180 degrees"
    private let notInDaylightPrompt = "You are not in daylight! Please turn off all artificial lights."
    private let isInDaylightPrompt = "Nice! You are in daylight."
    
    static func storyboardInstance() -> AssessFaceController? {
        let className = String(describing: AssessFaceController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? AssessFaceController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.previewView.rotation = .rotate180Degrees
        
        self.instructions.text = self.unknownPrompt
        self.instructions.textColor = UIColor.systemRed
        
        self.notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        self.notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.setupVideoCaptureSession()
        
        self.captureSessionQueue.async {
            self.captureSession.startRunning()
        }
    }
    
    @objc private func appMovedToBackground() {
        self.envObserver?.stopMotionUpdates()
        self.previewView.image = nil
    }
    
    @objc private func appMovedToForeground() {
        self.envObserver?.startMotionUpdates()
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
    
    // back allowes user to go back to previous view controller.
    @IBAction func back() {
        self.notifCenter.removeObserver(self)
        self.envObserver?.stopMotionUpdates()
        self.captureSession.stopRunning()
        self.previewView.image = nil
        self.dismiss(animated: true)
    }
}

extension AssessFaceController: EnvObserverDelegate {
    func daylightUpdated(isDaylight: Bool) {
        self.handleDayLightUpdate(isDaylight: isDaylight)
    }
    
    private func handleDayLightUpdate(isDaylight: Bool) {
        self.stateQueue.async {
            if self.state == .Unknown {
                return
            }
            if isDaylight {
                self.state = .IsInDaylight
            } else {
                self.state = .NotInDaylight
            }
            
            DispatchQueue.main.async {
                if isDaylight {
                    self.instructions.text = self.isInDaylightPrompt
                    self.instructions.textColor = UIColor.systemGreen
                } else {
                    self.instructions.text = self.notInDaylightPrompt
                    self.instructions.textColor = UIColor.systemRed
                }
            }
        }
    }
    
    func notifyISOUpdate(newISO: Int) {
        DispatchQueue.main.async {
            self.isoLabel.text = "ISO:" + String(newISO)
        }
    }
    
    func notifyTempUpdate(newTemp: Int) {
        DispatchQueue.main.async {
            self.tempLabel.text = String(newTemp) + "K"
        }
    }
    
    func notifyExposureUpdate(newExpsosure: Int) {
        DispatchQueue.main.async {
            self.exposureLabel.text = "E:" + String(newExpsosure)
        }
    }
    
    func notifyProgress(progress: Float) {
        DispatchQueue.main.async {
            self.progressView.setProgress(progress, animated: true)
        }
    }
    
    func notifyLightingTestResults(isIndoors: Bool, isDayLight: Bool, isGoodISO: Bool, isGoodExposure: Bool) {
        DispatchQueue.main.async {
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
        
        handleFaceFound()
        self.previewView.image = rgbImage
        
        if isPhoneTooClose(faceDepth: faceDepth) {
            // Wait for user to move phone further away.
            return
        }
    }
    
    // handleFaceFound starts turn around process once face is found for the first time.
    private func handleFaceFound() {
        if self.previewView.image != nil {
            return
        }
        self.stateQueue.async {
            self.state = .StartTurnAround
            self.envObserver?.observeLighting(device: self.cameraDevice, vc: self)
            self.envObserver?.startMotionUpdates()
        }
        DispatchQueue.main.async {
            self.instructions.text = self.turnAroundPrompt
            self.instructions.textColor = UIColor.systemIndigo
            self.instructions.blink()
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

extension UILabel {
    func blink() {
        UIView.animate(withDuration: 0.8,
          delay:0.0,
          options:[.allowUserInteraction, .curveEaseInOut, .autoreverse, .repeat],
          animations: { self.alpha = 0 },
          completion: nil)
    }
    
    func stopBlink() {
        self.layer.removeAllAnimations()
        self.alpha = 1
    }
}
