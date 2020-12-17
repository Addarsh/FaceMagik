//
//  StandupController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/15/20.
//

import UIKit

class StandupController: UIViewController {
    
    static func storyboardInstance() -> StandupController? {
        let className = String(describing: StandupController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? StandupController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    // back allows user to go back to previous view controller.
    @IBAction func back() {
        self.dismiss(animated: true)
    }
    
    // done allows user to progress to next view controller.
    @IBAction func done() {
        guard let vc = PhoneOrientationController.storyboardInstance() else {
            return
        }
        vc.modalPresentationStyle = .fullScreen
        self.present(vc, animated: true)
    }
}
