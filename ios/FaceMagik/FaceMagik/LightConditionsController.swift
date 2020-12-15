//
//  LightConditionsController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/13/20.
//

import UIKit

class LightConditionsController: UIViewController {
    static func storyboardInstance() -> LightConditionsController? {
        let className = String(describing: LightConditionsController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? LightConditionsController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    // userNotReady indicates user is not ready to test environmental lighting conditions.
    @IBAction func userNotReady() {
        self.dismiss(animated: true)
    }
}
