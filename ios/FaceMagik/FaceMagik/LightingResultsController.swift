//
//  BadLightingController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/19/20.
//

import UIKit

class LightingResultsController: UIViewController {
    @IBOutlet var titleMsg: UITextView!
    @IBOutlet var indoorsImage: UIImageView!
    @IBOutlet var daylightImage: UIImageView!
    @IBOutlet var wellLitImage: UIImageView!
    @IBOutlet var indoorErrorView: UITextView!
    @IBOutlet var dayLightErrorView: UITextView!
    @IBOutlet var wellLitErrorView: UITextView!
    @IBOutlet var button: UIButton!
    
    private static let dayLightErrorMsg: String = "It looks like the primary light source is not natural light. Please turn off any artificial light sources like LED light bulbs or lamps. If it's night time, please try again during the day anytime between sunrise to sunset."
    private static let lowLightErrorMsg: String = "There isn't adequate light in your surroundings. For best results, please move to a better lit area and try again. For example, moving closer to a window can help."
    private static let brightLightErrorMsg: String = "There is too much light in your surroundings. It looks like you may be outdoors or very close to an open window or door. For best results, please be indoors and little 5-6 steps away from the an open window or door."
    private static let resultsErrorMsg: String = "Lighting conditions are not appropriate."
    private static let resultsGoodMsg: String = "Lighting conditions are good!"
    private static let unwindSegueIdentifier: String = "StandupController"
    private static let continueButtonTitle = "Continue"
    private static let tryAgainTitle = "Try Again"
    private static let greenCheckBoxImage: String = "green-checkmark.png"
    
    public var isIndoors: Bool = false
    public var isDayLight: Bool = false
    public var isGoodISO: Bool = false
    public var isGoodExposure: Bool = false
    private var isGoodLighting: Bool = false
    
    static func storyboardInstance() -> LightingResultsController? {
        let className = String(describing: LightingResultsController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? LightingResultsController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        if self.isGoodExposure {
            self.indoorsImage.image = UIImage(named: LightingResultsController.greenCheckBoxImage)
        } else {
            self.indoorErrorView.text = LightingResultsController.brightLightErrorMsg
        }
        if self.isDayLight {
            self.daylightImage.image = UIImage(named: LightingResultsController.greenCheckBoxImage)
        } else {
            self.dayLightErrorView.text = LightingResultsController.dayLightErrorMsg
        }
        if self.isGoodISO {
            self.wellLitImage.image = UIImage(named: LightingResultsController.greenCheckBoxImage)
        } else {
            self.wellLitErrorView.text = LightingResultsController.lowLightErrorMsg
        }
        
        if self.isDayLight && self.isGoodISO && self.isGoodExposure {
            self.titleMsg.text = LightingResultsController.resultsGoodMsg
            self.button.setTitle(LightingResultsController.continueButtonTitle, for: .normal)
            self.isGoodLighting = true
        } else {
            self.titleMsg.text = LightingResultsController.resultsErrorMsg
            self.button.setTitle(LightingResultsController.tryAgainTitle, for: .normal)
            self.isGoodLighting = false
        }
    }
    
    // next allows user to go to next view controller depending on lighting conditions.
    @IBAction func next() {
        if testMode || self.isGoodLighting {
            guard let vc = RemindPhoneOrientationController.storyboardInstance() else {
                return
            }
            vc.modalPresentationStyle = .fullScreen
            self.present(vc, animated: true)
        } else {
            self.performSegue(withIdentifier: LightingResultsController.unwindSegueIdentifier, sender: self)
        }
    }
}
