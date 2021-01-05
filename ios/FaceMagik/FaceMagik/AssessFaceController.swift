//
//  AssessFaceController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/25/20.
//

import UIKit
import Photos

protocol FaceProcessor {
    func startDetection(vc: FaceProcessorDelegate?)
    func getDevice() -> AVCaptureDevice
    func getFaceMask() -> CIImage?
    func stop()
    func resume()
}

protocol FaceProcessorDelegate {
    func firstFrame()
    func frameUpdated(rgbImage: CIImage, faceDepth: Float)
}

protocol EnvObserver {
    func observeLighting(device: AVCaptureDevice?, vc: EnvObserverDelegate?)
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
    @IBOutlet weak private var previewView: PreviewMetalView!
    
    private let notifCenter = NotificationCenter.default
    var faceDetector: FaceProcessor?
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
    
        self.resetState()
        
        self.notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        self.notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.faceDetector?.startDetection(vc: self)
    }
    
    @objc private func appMovedToBackground() {
        self.envObserver?.stopMotionUpdates()
        self.faceDetector?.stop()
        self.previewView.image = nil
    }
    
    @objc private func appMovedToForeground() {
        self.resetState()
        self.envObserver?.startMotionUpdates()
        self.faceDetector?.resume()
    }
    
    private func resetState() {
        self.instructions.stopBlink()
        self.state = .Unknown
        self.instructions.text = self.unknownPrompt
        self.instructions.textColor = UIColor.systemRed
    }
    
    // back allowes user to go back to previous view controller.
    @IBAction func back() {
        self.notifCenter.removeObserver(self)
        self.envObserver?.stopMotionUpdates()
        self.faceDetector?.stop()
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

extension AssessFaceController: FaceProcessorDelegate {
    func firstFrame() {
        self.stateQueue.async {
            self.state = .StartTurnAround
            self.envObserver?.observeLighting(device: self.faceDetector?.getDevice(), vc: self)
            self.envObserver?.startMotionUpdates()
        }
        DispatchQueue.main.async {
            self.instructions.text = self.turnAroundPrompt
            self.instructions.textColor = UIColor.systemIndigo
            self.instructions.blink()
        }
    }
    
    func frameUpdated(rgbImage: CIImage, faceDepth: Float) {
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
