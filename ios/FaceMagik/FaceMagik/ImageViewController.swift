//
//  ImageViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 10/23/20.
//
import UIKit
import Photos

class ImageViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    var image: UIImage!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized else { return }
        }
        
        imageView.image = image
    }
    
    @IBAction func didCancel() {
        dismissViewController()
    }
    
    @IBAction func didSave() {
        guard let data = pngData() else {
            print ("Could not convert UIImage to PNG Data")
            return
        }
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized else { return }
            PHPhotoLibrary.shared().performChanges({
                let creationRequest = PHAssetCreationRequest.forAsset()
            creationRequest.addResource(with: .photo, data: data, options: nil)
            }, completionHandler: { success, error in
                if (!success) {
                    print ("Could not save photo to library with \(String(describing: error))")
                }
                DispatchQueue.main.async {
                    self.dismissViewController()
                }
            })
        }
    }
    
    func pngData() -> Data? {
        guard let image = imageView.image else {
            print ("UIImage not found")
            return nil
        }
        
        guard let ciImage = CIImage(image: image) else {
            print ("could not convert uiImage to ciImage")
            return nil
        }
        
        let newciImage = ciImage.oriented(forExifOrientation: Int32(CGImagePropertyOrientation.right.rawValue))
        
        let context = CIContext(options: [CIContextOption.useSoftwareRenderer: true])
        guard let data = context.pngRepresentation(of: newciImage, format: CIFormat.ARGB8, colorSpace: CGColorSpace(name: CGColorSpace.displayP3)!) else {
            return nil
        }
        return data
    }
    
    func dismissViewController() {
        guard let navController = self.navigationController else {
            print ("Navigation controller missing")
            return
        }
        navController.popViewController(animated: true)
        dismiss(animated: true, completion: nil)
    }
}
