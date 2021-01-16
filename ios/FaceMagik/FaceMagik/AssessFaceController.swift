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
    func stop()
    func resume()
}

protocol FaceProcessorDelegate {
    func firstFrame()
    func frameUpdated(faceProperties: FaceProperties)
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
    func notifyHeading(heading: Int)
    func motionUpdating()
    func motionUpdateComplete()
    func badColorTemperature()
    func possiblyOutdoors()
    func tooBright()
}

protocol AssessFaceControllerDelegate {
    func handleUpdatedHeading(heading: Int)
    func handleUpdatedImageValues(leftCheekPercentValue: Int, rightCheekPercentValue: Int)
}

class AssessFaceController: UIViewController {
    @IBOutlet private var isoLabel: UILabel!
    @IBOutlet private var tempLabel: UILabel!
    @IBOutlet private var exposureLabel: UILabel!
    @IBOutlet private var instructions: UILabel!
    @IBOutlet weak private var previewView: PreviewMetalView!
    @IBOutlet private var resultLabel: UILabel!
    @IBOutlet private var leftCheekValueLabel: UILabel!
    @IBOutlet private var rightCheekValueLabel: UILabel!
    
    private let notifCenter = NotificationCenter.default
    var faceDetector: FaceProcessor?
    var envObserver: EnvObserver?
    var skinAnalyzerDelegate: AssessFaceControllerDelegate?
    var stateMgr: StateManager?
    private var phoneTooCloseAlert: AlertViewController?

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
        
        self.skinAnalyzerDelegate = SkinToneAnalyzer()
        self.stateMgr = StateManager()
        
        self.previewView.rotation = .rotate180Degrees
        self.previewView.mirroring = true
        
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
        self.stateMgr?.updateState(state: StateManager.State.Unknown)
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
    
    func notifyHeading(heading: Int) {
        if self.stateMgr?.getState() == StateManager.State.StartTurnAround {
            self.skinAnalyzerDelegate?.handleUpdatedHeading(heading: heading)
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
        self.stateMgr?.updateState(state: StateManager.State.TurnAroundComplete)
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
    
    func tooBright() {
        // Delay.
        // Start movement towards direction of light.
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
        self.stateMgr?.updateState(state: StateManager.State.StartTurnAround)
        self.envObserver?.observeLighting(device: self.faceDetector?.getDevice(), vc: self)
        self.envObserver?.startMotionUpdates(range: 180)
        
        DispatchQueue.main.async {
            self.instructions.text = self.turnAroundPrompt
            self.instructions.textColor = UIColor.systemIndigo
            self.instructions.blink()
        }
    }
    
    func frameUpdated(faceProperties: FaceProperties) {
        self.previewView.image = CIImageHelper.overlayMask(image: faceProperties.image, mask: CIImageHelper.bitwiseXor(firstMask: faceProperties.leftCheekMask, secondMask: faceProperties.rightCheekMask)!)
        
        DispatchQueue.main.async {
            self.leftCheekValueLabel.text = String(faceProperties.leftCheekPercentValue)
            self.rightCheekValueLabel.text = String(faceProperties.rightCheekPercentValue)
        }
        
        if self.stateMgr?.getState() == StateManager.State.StartTurnAround {
            self.skinAnalyzerDelegate?.handleUpdatedImageValues(leftCheekPercentValue: faceProperties.leftCheekPercentValue, rightCheekPercentValue: faceProperties.rightCheekPercentValue)
        }
        
        if isPhoneTooClose(faceDepth: faceProperties.faceDepth) {
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
