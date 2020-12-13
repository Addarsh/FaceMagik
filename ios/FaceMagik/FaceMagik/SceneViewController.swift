//
//  SceneViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/7/20.
//

import UIKit
import Photos
import Vision

enum SceneType: Int, Codable {
    case Indoors = 1
    case Outdoors = 2
}

class SceneViewController: UIViewController {
    @IBOutlet private var previewLayer: PreviewView!
    @IBOutlet private var sceneLabel: UILabel!
    @IBOutlet private var confidenceLabel: UILabel!
    
    // AVCaptureSession variables.
    @objc var cameraDevice: AVCaptureDevice!
    var sessionQueue: DispatchQueue!
    var captureSession =  AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    private let videoOutputQueue = DispatchQueue(label: "com.example.FaceMagik.videoOutputQueue")
    var visionRequests: [VNCoreMLRequest] = []
    var sceneMappings: [String: SceneType] = [:]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let notifCenter = NotificationCenter.default
        notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        guard let mappings = self.readSceneMappings() else {
            print ("Error! Could not read scene mappings")
            return
        }
        self.sceneMappings = mappings
            
        // Setup video capture session.
        self.setupVideoCaptureSession()
        
        self.previewLayer.videoPreviewLayer.session = self.captureSession
        
        self.prepareVisionRequest()
        
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
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        if !self.captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.startRunning()
            }
        }
    }
    
    @objc func appMovedToBackground() {
        if self.captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.stopRunning()
            }
        }
    }
    
    @objc func appMovedToForeground() {
        if !self.captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.startRunning()
            }
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if self.captureSession.isRunning {
            self.captureSession.stopRunning()
        }
    }
    
    // prepareVisionRequest prepares a vision request using already existing CoreML mode.
    func prepareVisionRequest() {
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
            if res.confidence > 0  && self.sceneMappings[res.identifier] != nil {
                DispatchQueue.main.async {
                    self.sceneLabel.text = String(describing: self.sceneMappings[res.identifier]!)
                    self.confidenceLabel.text = String(Int(res.confidence * 100))
                }
            }
        }
        
        self.visionRequests = [vRequest]
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
}

extension SceneViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
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
