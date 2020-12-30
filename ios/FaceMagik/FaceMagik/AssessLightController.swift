//
//  SceneViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/7/20.
//

import UIKit
import Photos
import Vision
import CoreMotion

enum SceneType: Int, Codable {
    case Indoors = 1
    case Outdoors = 2
}

struct SensorValues {
    var iso: Int
    var exposure: Int
    var temp: Int
    var sceneType: SceneType
}

class AssessLightController: UIViewController {
    @IBOutlet private var instructions: UITextView!
    @IBOutlet private var previewLayer: PreviewView!
    @IBOutlet private var progressView: UIProgressView!
    @IBOutlet var scenePercentLabel: UILabel!
    @IBOutlet var tempPercentLabel: UILabel!
    @IBOutlet var isoPercentLabel: UILabel!
    @IBOutlet var expPercentLabel: UILabel!
    @IBOutlet var colorLabel: UILabel!
    @IBOutlet var isoLabel: UILabel!
    @IBOutlet var exposureLabel: UILabel!
    
    private var exposureObservation: NSKeyValueObservation?
    private var tempObservation: NSKeyValueObservation?
    private var isoObservation: NSKeyValueObservation?
    private var currTemp: Int = 0
    private var currISO: Int = 0
    private var currExposure: Int = 0
    private var currSceneType: SceneType = .Indoors
    private var sensorMap: [Int: SensorValues] = [:]
    private let serialQueue = DispatchQueue(label: "Serial Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private let notifCenter = NotificationCenter.default
    static private let indoorPercentThreshold = 70
    static private let expPercentThreshold = 60
    static private let isoPerentThreshold = 70
    static private let colorTempThreshold = 70
    
    // Core Motion variables.
    private let motionManager = CMMotionManager()
    private var motionQueue = OperationQueue()
    static private let motionFrequency = 1.0/30.0
    static private let totalHeadingVals = 320
    
    // AVCaptureSession variables.
    @objc var cameraDevice: AVCaptureDevice!
    private var sessionQueue: DispatchQueue!
    private var captureSession =  AVCaptureSession()
    private let captureSessionQueue = DispatchQueue(label: "vision request queue", qos: .userInteractive, attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoOutputQueue = DispatchQueue(label: "com.example.FaceMagik.videoOutputQueue")
    private var visionRequests: [VNCoreMLRequest] = []
    private var sceneMappings: [String: SceneType] = [:]
    
    static func storyboardInstance() -> AssessLightController? {
        let className = String(describing: AssessLightController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? AssessLightController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        if !self.motionManager.isDeviceMotionAvailable {
            print ("Device motion unavaible! Error!")
            return
        }
        
        self.instructions.text = "Rotate slowly until the progress bar completes"
        
        self.notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        self.notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.captureSessionQueue.async {
            self.prepareVisionRequest()
        }
        
        self.setupVideoCaptureSession()
        self.previewLayer.videoPreviewLayer.session = self.captureSession
        self.observeDevice()
        
        self.captureSessionQueue.async {
            self.captureSession.startRunning()
            self.startMotionUpdates()
        }
    }
    
    // back allows user to go back to previous veiwcontroller.
    @IBAction func back() {
        self.notifCenter.removeObserver(self)
        self.motionManager.stopDeviceMotionUpdates()
        self.previewLayer.videoPreviewLayer.session = nil
        self.captureSession.stopRunning()
        self.dismiss(animated: true)
    }
    
    // readSceneMappings reads scene mappings data from JSON file.
    func readSceneMappings() -> [String: SceneType]? {
        if let path = Bundle.main.path(forResource: "SceneMapping", ofType: "json") {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
                let d = try JSONSerialization.jsonObject(with: data, options: .mutableContainers) as? [String:String]
                return d!.mapValues{SceneType(rawValue: Int($0)!)!}
            } catch let error as NSError {
                print ("Failed to read scene mapping file with error: \(error)")
            }
        }
        return nil
    }
    
    @objc func appMovedToBackground() {
        if self.motionManager.isDeviceMotionActive {
            self.motionManager.stopDeviceMotionUpdates()
        }
        self.previewLayer.videoPreviewLayer.session = nil
    }
    
    @objc func appMovedToForeground() {
        if !self.captureSession.isRunning {
            self.previewLayer.videoPreviewLayer.session = self.captureSession
            self.captureSession.startRunning()
        }
            
        if !self.motionManager.isDeviceMotionActive {
            self.startMotionUpdates()
        }
    }
    
    // prepareVisionRequest prepares a vision request using already existing CoreML mode.
    func prepareVisionRequest() {
        guard let mappings = self.readSceneMappings() else {
            print ("Error! Could not read scene mappings")
            return
        }
        self.sceneMappings = mappings
        
        guard let model = try? VNCoreMLModel(for: GoogLeNetPlaces(configuration: MLModelConfiguration.init()).model) else {
            print ("Failed to intialize Core ML Model")
            return
        }
        
        let vRequest = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation] else {
                print ("Failed to retrieve results from CoreML classifier")
                return
            }
            let res = results.first!
            if let sceneType = self.sceneMappings[res.identifier] {
                self.currSceneType = sceneType
            }
        }
        
        self.visionRequests = [vRequest]
    }
    
    // observeDevice observes the exposure duration and color temperature of current device.
    func observeDevice() {
        // Start observing camera device exposureDuration.
        self.exposureObservation = observe(\.self.cameraDevice.exposureDuration, options: .new){
            object, change in
            guard let newVal = change.newValue else {
                return
            }
            self.serialQueue.async {
                self.currExposure = Int(1/(newVal.seconds))
            }
            DispatchQueue.main.async {
                self.exposureLabel.text = "E:" + String(Int(1/(newVal.seconds)))
            }
        }
        
        // Start observing camera device white balance gains.
        self.tempObservation = observe(\.self.cameraDevice.deviceWhiteBalanceGains, options: .new){
            obj, chng in
            let temp = self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature
            self.serialQueue.async {
                self.currTemp = Int(temp)
            }
            DispatchQueue.main.async {
                self.colorLabel.text = String(Int(temp))
            }
        }
        
        // Start observing camera device white balance gains.
        self.isoObservation = observe(\.self.cameraDevice.iso, options: .new){
            obj, change in
            guard let newVal = change.newValue else {
                return
            }
            self.serialQueue.async {
                self.currISO = Int(newVal)
            }
            DispatchQueue.main.async {
                self.isoLabel.text = String(Int(newVal))
            }
        }
    }
    
    // setupVideoCaptureSession sets up a capture session to capture video.
    func setupVideoCaptureSession() {
        self.captureSession.beginConfiguration()
        
        // Add capture session input.
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
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
        self.videoOutput.setSampleBufferDelegate(self, queue: self.videoOutputQueue)
        
        self.captureSession.commitConfiguration()
    }
    
    // startMotionUpdates starts to receive motion updates from motion manager.
    func startMotionUpdates() {
        self.motionManager.deviceMotionUpdateInterval = AssessLightController.motionFrequency
        self.motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: self.motionQueue, withHandler: { (data, error) in
            guard let validData = data else {
                return
            }
            let heading = Int(validData.heading)
            self.serialQueue.async {
                if self.sensorMap.keys.count >= AssessLightController.totalHeadingVals {
                    // Completed sensor data collection.
                    return
                }
                if self.sensorMap[heading] != nil {
                    return
                }
                self.sensorMap[heading] = SensorValues(iso: self.currISO, exposure: self.currExposure, temp: self.currTemp, sceneType: self.currSceneType)
                
                let kCount = self.sensorMap.keys.count
                DispatchQueue.main.async {
                    self.progressView.setProgress(Float(kCount)/Float(AssessLightController.totalHeadingVals), animated: true)
                }
                if kCount == AssessLightController.totalHeadingVals {
                    self.validateEnv()
                }
            }
        })
    }
    
    // validateEnv validates if environment is suitable for pictures using existing sensor values.
    // Returns true/false values for if env is indoors, in daylight and well lit respectively.
    func validateEnv() {
        // Check indoor/outdoor label ratio.
        var numIndoors = 0
        var numVisibleColorTemp = 0
        var numGoodISO = 0
        var numGoodExposure = 0
        for (_, readouts) in self.sensorMap {
            if readouts.sceneType == .Indoors {
                numIndoors += 1
            }
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
        
        let indoorPercent = Int((Float(numIndoors)/Float(self.sensorMap.keys.count))*100.0)
        let colorTempPercent = Int((Float(numVisibleColorTemp)/Float(self.sensorMap.keys.count))*100.0)
        let isoPercent = Int((Float(numGoodISO)/Float(self.sensorMap.keys.count))*100.0)
        let expPercent = Int((Float(numGoodExposure)/Float(self.sensorMap.keys.count))*100.0)
        
        DispatchQueue.main.async {
            self.scenePercentLabel.text = String(indoorPercent) + "%"
            self.tempPercentLabel.text = String(colorTempPercent) + "%"
            self.isoPercentLabel.text = String(isoPercent) + "%"
            self.expPercentLabel.text =  String(expPercent) + "%"
            self.goToNextController(isIndoors: indoorPercent >= AssessLightController.indoorPercentThreshold ? true : false, isDayLight: colorTempPercent >= AssessLightController.colorTempThreshold ? true : false, isGoodISO: isoPercent >= AssessLightController.isoPerentThreshold ? true : false, isGoodExposure: expPercent >= AssessLightController.expPercentThreshold ? true : false)
        }
        
    }
    
    // goToNextController goes to next view controller.
    private func goToNextController(isIndoors: Bool, isDayLight: Bool, isGoodISO: Bool, isGoodExposure: Bool) {
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

extension AssessLightController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to obtain a CVPixelBuffer for the current output frame.")
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let handler = VNImageRequestHandler(ciImage: ciImage)
        
        do {
            try handler.perform(self.visionRequests)
        } catch let error as NSError {
            print ("Failed to classify image with error: \(error)")
        }
    }
}
