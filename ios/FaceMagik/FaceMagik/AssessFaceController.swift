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
    func startMotionUpdates(range: Int)
    func stopMotionUpdates()
}

protocol EnvObserverDelegate {
    func notifyISOUpdate(newISO: Int)
    func notifyExposureUpdate(newExpsosure: Int)
    func notifyTempUpdate(newTemp: Int)
    func motionUpdating()
    func motionUpdateComplete()
    func badColorTemperature()
    func possiblyOutdoors()
}

class AssessFaceController: UIViewController {
    enum State {
        case Unknown
        case StartTurnAround
        case TurnAroundComplete
    }
    
    @IBOutlet private var isoLabel: UILabel!
    @IBOutlet private var tempLabel: UILabel!
    @IBOutlet private var exposureLabel: UILabel!
    @IBOutlet private var instructions: UILabel!
    @IBOutlet weak private var previewView: PreviewMetalView!
    @IBOutlet private var resultLabel: UILabel!
    
    private let notifCenter = NotificationCenter.default
    var faceDetector: FaceProcessor?
    var envObserver: EnvObserver?
    private var phoneTooCloseAlert: AlertViewController?
    
    private var state: State = .Unknown
    private let stateQueue = DispatchQueue(label: "State Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private let unknownPrompt = "Waiting to detect face"
    private let turnAroundPrompt = "Turn Around 180 degrees"
    private let keepTurningPrompt = "Keep Turning..."
    private let stopPrompt = "Stop"
    
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
    
    func motionUpdating() {
        DispatchQueue.main.async {
            self.instructions.text = self.keepTurningPrompt
            self.instructions.textColor = UIColor.systemIndigo
        }
    }
    
    func motionUpdateComplete() {
        self.envObserver?.stopMotionUpdates()
        DispatchQueue.main.async {
            self.instructions.stopBlink()
            self.instructions.text = self.stopPrompt
            self.instructions.textColor = UIColor.systemRed
        }
        
    }
    
    func badColorTemperature() {
        DispatchQueue.main.async {
            self.resultLabel.text = "Bad color"
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            guard let vc = BadColorTemperature.storyboardInstance() else {
                return
            }
            self.present(vc, animated: true)
        }
    }
    
    func possiblyOutdoors() {
        DispatchQueue.main.async {
            self.resultLabel.text = "Outdoors"
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            guard let vc = PossiblyOutsideError.storyboardInstance() else {
                return
            }
            self.present(vc, animated: true)
        }
    }
    
    func displayError(isIndoors: Bool, isDayLight: Bool, isGoodISO: Bool, isGoodExposure: Bool) {
        DispatchQueue.main.async {
            guard let vc = LightingResultsController.storyboardInstance() else {
                return
            }
            vc.isIndoors = isIndoors
            vc.isDayLight = isDayLight
            vc.isGoodISO = isGoodISO
            vc.isGoodExposure = isGoodExposure
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
            self.envObserver?.startMotionUpdates(range: 180)
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
