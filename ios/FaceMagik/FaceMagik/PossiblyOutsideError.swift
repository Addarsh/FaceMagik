//
//  PossiblyOutsideError.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/9/21.
//

import UIKit

class PossiblyOutsideError: UIViewController {
    
    static func storyboardInstance() -> PossiblyOutsideError? {
        let className = String(describing: PossiblyOutsideError.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? PossiblyOutsideError
    }
    
    // tryAgain allowes user to go back to previous view controller.
    @IBAction func tryAgain() {
        self.dismiss(animated: true)
    }
}
