//
//  ViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 10/16/20.
//

import UIKit
import Photos

class ViewController: UIViewController {
    let picker = UIImagePickerController()
    var image: UIImageView!
    @IBOutlet var overlayView: UIView!

    override func viewDidLoad() {
        super.viewDidLoad()

        startPicker()
    }

    func startPicker() {
        picker.sourceType = .camera
        picker.cameraDevice = .rear
        picker.showsCameraControls = false
        picker.delegate = self
        picker.cameraOverlayView = overlayView
        
        picker.cameraViewTransform = CGAffineTransform(translationX: 0, y: 120)
        
        present(picker, animated: true)
    }
    
    @IBAction func switchCamera() {
        if picker.cameraDevice == .rear {
            picker.cameraDevice = .front
        }else {
            picker.cameraDevice = .rear
        }
    }
}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
}

