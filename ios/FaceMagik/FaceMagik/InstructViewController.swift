//
//  InstructViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 11/26/20.
//

import UIKit


class InstructViewController: UIViewController {
    private let segueIdentifier = "portaitView"
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    // goToNext performs segue to next view controller.
    @IBAction func goToNext() {
        performSegue(withIdentifier: self.segueIdentifier, sender: nil)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        super.prepare(for: segue, sender: sender)

        if let destVC = segue.destination as? EnvViewController {
            destVC.modalPresentationStyle = .fullScreen
        }
    }
}
