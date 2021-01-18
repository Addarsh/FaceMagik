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

protocol MotionObserver {
    func startMotionUpdates(delegate: MotionObserverDelegate?)
    func stopMotionUpdates()
    func ensureUserRotates(range: Int)
}

protocol MotionObserverDelegate {
    func updatedHeading(heading: Int)
    func userHasStartedRotating()
    func wrongRotationDirection()
    func rotationComplete()
}

protocol LightingObserver {
    func startObserving(device: AVCaptureDevice?, delegate: LightingObserverDelegate?)
    func stopObserving()
}

protocol LightingObserverDelegate {
    func updatedISO(iso: Int)
    func updatedExposure(exposure: Int)
    func updatedColorTemp(temp: Int)
}

protocol EnvObserver {
    func updatedHeading(heading: Int)
    func updatedISO(iso: Int)
    func updatedExposure(exposure: Int)
    func updatedColorTemp(temp: Int)
    func observeLighting(delegate: EnvObserverDelegate?)
    func processLighting()
}

protocol EnvObserverDelegate {
    func badColorTemperature()
    func possiblyOutdoors()
    func tooBright()
}

protocol AssessFaceControllerDelegate {
    func handleUpdatedHeading(heading: Int)
    func handleUpdatedImageValues(leftCheekPercentValue: Int, rightCheekPercentValue: Int)
    func estimatePrimaryLightDirection() -> Int
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
    @IBOutlet private var cheekValueDiffLabel: UILabel!
    @IBOutlet private var headingLabel: UILabel!
    
    private let notifCenter = NotificationCenter.default
    private var cameraDevice: AVCaptureDevice?
    var faceDetector: FaceProcessor?
    var envObserver: EnvObserver?
    var skinAnalyzerDelegate: AssessFaceControllerDelegate?
    var lightingObserver: LightingObserver?
    var motionObserver: MotionObserver?
    var stateMgr: StateManager?
    private var phoneTooCloseAlert: AlertViewController?

    private let unknownPrompt = "Waiting to detect face"
    private let turnAroundPrompt = "Turn Around 360 degrees"
    private let keepTurningPrompt = "Keep Turning..."
    private let turningWrongDirectionPrompt = "Wrong direction! Turn other way"
    private let stopPrompt = "Stop"
    
    static func storyboardInstance() -> AssessFaceController? {
        let className = String(describing: AssessFaceController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? AssessFaceController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.stateMgr = StateManager()
        
        self.previewView.rotation = .rotate180Degrees
        self.previewView.mirroring = true
        
        self.resetState()
        
        self.notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        self.notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.motionObserver?.startMotionUpdates(delegate: self)
        self.faceDetector?.startDetection(vc: self)
    }
    
    @objc private func appMovedToBackground() {
        self.motionObserver?.stopMotionUpdates()
        self.lightingObserver?.stopObserving()
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
        self.motionObserver?.stopMotionUpdates()
        self.lightingObserver?.stopObserving()
        self.faceDetector?.stop()
        self.previewView.image = nil
        self.dismiss(animated: true)
    }
}

extension AssessFaceController: MotionObserverDelegate {
    func updatedHeading(heading: Int) {
        if self.stateMgr?.getState() == StateManager.State.StartTurnAround {
            self.envObserver?.updatedHeading(heading: heading)
            self.skinAnalyzerDelegate?.handleUpdatedHeading(heading: heading)
        }
        DispatchQueue.main.async {
            self.headingLabel.text = String(heading)
        }
    }
    
    func userHasStartedRotating() {
        DispatchQueue.main.async {
            self.instructions.text = self.keepTurningPrompt
            self.instructions.textColor = UIColor.systemIndigo
        }
    }
    
    func wrongRotationDirection() {
        DispatchQueue.main.async {
            self.instructions.text = self.turningWrongDirectionPrompt
            self.instructions.textColor = UIColor.systemRed
        }
    }
    
    func rotationComplete() {
        let primaryLightDirection = self.skinAnalyzerDelegate?.estimatePrimaryLightDirection()
        self.stateMgr?.updateState(state: StateManager.State.TurnAroundComplete)
        self.envObserver?.processLighting()
        DispatchQueue.main.async {
            self.instructions.stopBlink()
            self.instructions.text = self.stopPrompt
            self.instructions.textColor = UIColor.systemRed
            
            // Primary Light Direction.
            self.resultLabel.text = String(primaryLightDirection ?? -1)
        }
    }
}

extension AssessFaceController: EnvObserverDelegate {
    func badColorTemperature() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            guard let vc = BadColorTemperature.storyboardInstance() else {
                return
            }
            self.present(vc, animated: true)
        }
    }
    
    func possiblyOutdoors() {
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

extension AssessFaceController: LightingObserverDelegate {
    func updatedISO(iso: Int) {
        self.envObserver?.updatedISO(iso: iso)
        DispatchQueue.main.async {
            self.isoLabel.text = "ISO:" + String(iso)
        }
    }
    
    func updatedExposure(exposure: Int) {
        self.envObserver?.updatedExposure(exposure: exposure)
        DispatchQueue.main.async {
            self.exposureLabel.text = "E:" + String(exposure)
        }
    }
    
    func updatedColorTemp(temp: Int) {
        self.envObserver?.updatedColorTemp(temp: temp)
        DispatchQueue.main.async {
            self.tempLabel.text = String(temp) + "K"
        }
    }
    
}

extension AssessFaceController: FaceProcessorDelegate {
    func firstFrame() {
        self.cameraDevice = self.faceDetector?.getDevice()
        self.stateMgr?.updateState(state: StateManager.State.StartTurnAround)
        self.motionObserver?.ensureUserRotates(range: 360)
        self.lightingObserver?.startObserving(device: self.cameraDevice, delegate: self)
        
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
            self.cheekValueDiffLabel.text = String(abs(faceProperties.leftCheekPercentValue - faceProperties.rightCheekPercentValue))
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
