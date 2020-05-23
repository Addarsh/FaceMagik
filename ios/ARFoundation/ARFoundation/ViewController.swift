//
//  ViewController.swift
//  ARFoundation
//
//  Created by Addarsh Chandrasekar on 1/20/20.
//  Copyright © 2020 Addarsh Chandrasekar. All rights reserved.
//

import UIKit
import SceneKit
import ARKit
import VideoToolbox
import Alamofire

class ViewController: UIViewController, ARSCNViewDelegate {
    @IBOutlet weak var captureButton: UIButton!
    @IBOutlet weak var lightIntensity: UILabel!
    @IBOutlet weak var ambientIntensity: UILabel!
    @IBOutlet weak var xDirection: UILabel!
    @IBOutlet weak var progressView: UIProgressView!
    @IBOutlet var sceneView: ARSCNView!
    
    private let concurrentPhotoQueue = DispatchQueue(label: "com.ARFoundation.photoqueue", attributes: .concurrent)
    
    private var lastCapturedImage: CVPixelBuffer?
    
    private var pngData: Data?
    
    private let uploadURL = "http://[2600:1700:f1b0:6400:e82b:b259:83c6:94cd]:8000/skin/"
    
    private var lastFaceAnchor: ARFaceAnchor?
    
    private var lastCamera: ARCamera?
    
    private var lastLightEstimate: ARDirectionalLightEstimate?
    
    private var lightDict: [String: [Float]]?
    
    private var vertices2D: [[Int]]?
    
    private var vertexNormals: [[Float]]?
    
    private var triangleIndices: [Int16]?
    
    // Moving average variables.
    private let mAvgInterval: Int = 20
    
    private var dir: [[Float]] = [[], [], []]
    private var mavgDir: [Float] = [0.0, 0.0, 0.0]
    private var dirCount: Int = 0
    
    private var primary: [Float] = []
    private var mavgPrimary: Float = 0.0
    private var primaryCount: Int = 0
    
    private var ambient: [Float] = []
    private var mavgAmbient: Float = 0.0
    private var ambientCount: Int = 0
    
    private var harmonicPrint: [Float] = [0, 0, 0]
    private var harmonicPrintCount: Int = 0

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        sceneView.delegate = self
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        guard ARFaceTrackingConfiguration.isSupported else {
          fatalError("Face tracking is not supported on this device")
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        configuration.worldAlignment = .gravity

        // Run the view's session
        sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }

    // MARK: - ARSCNViewDelegate
    
/*
    // Override to create and configure nodes for anchors added to the view's session.
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        let node = SCNNode()
     
        return node
    }
*/
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user
        
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        // Inform the user that the session has been interrupted, for example, by presenting an overlay
        
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required
        
    }
    
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        guard let device = sceneView.device else {
            return nil
        }
        let faceGeometry = ARSCNFaceGeometry(device: device)
        let node = SCNNode(geometry: faceGeometry)
        node.geometry?.firstMaterial?.fillMode = .lines
        return node
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let faceAnchor = anchor as? ARFaceAnchor, let faceGeometry = node.geometry as? ARSCNFaceGeometry else {
            return
        }
        
        faceGeometry.update(from: faceAnchor.geometry)
        
        // Save last captured image.
        concurrentPhotoQueue.async(flags: .barrier) {
            [weak self] in
            guard let self = self else {
                return
            }
            
            self.lastFaceAnchor = faceAnchor
            self.lastCamera = self.sceneView.session.currentFrame?.camera
            self.lastCapturedImage = self.sceneView.session.currentFrame?.capturedImage
            self.lastLightEstimate = self.sceneView.session.currentFrame?.lightEstimate as? ARDirectionalLightEstimate
            if self.lastLightEstimate == nil {
                return
            }
            
            self.movingAverageXintensity()
            self.movingAveragePrimary()
            self.movingAverageAmbient()
            
            
            //self.printHarmonics(lightEstimate: lighstEstimate)
            DispatchQueue.main.async {
                self.lightIntensity.text = String(describing: Int(self.mavgPrimary))
                self.ambientIntensity.text = String(describing: Int(self.mavgAmbient))
                self.xDirection.text = String(describing: self.mavgDir[0])
                self.lightIntensity.setNeedsDisplay()
                self.ambientIntensity.setNeedsDisplay()
                self.xDirection.setNeedsDisplay()
            }
        }
    }
    
    // Calculate moving average direction for primaryIntensity
    // in X direction in face coordinates.
    func movingAverageXintensity() {
        let dirval = self.toFaceCords(self.lastLightEstimate!.primaryLightDirection, self.lastFaceAnchor!)
        if self.dirCount < self.mAvgInterval {
            self.dir[0].append(dirval[0])
            self.dir[1].append(dirval[1])
            self.dir[2].append(dirval[2])
        } else {
            self.dir[0][self.dirCount % self.mAvgInterval] = dirval[0]
            self.dir[1][self.dirCount % self.mAvgInterval] = dirval[1]
            self.dir[2][self.dirCount % self.mAvgInterval] = dirval[2]
        }
        self.dirCount += 1
        self.mavgDir = [self.dir[0].average(), self.dir[1].average(), self.dir[2].average()]
    }
    
    // Calculate moving average intensity for primaryIntensity.
    func movingAveragePrimary() {
        let pval = Float(self.lastLightEstimate!.primaryLightIntensity)/1000.0
        if self.primaryCount < self.mAvgInterval {
            self.primary.append(pval)
        } else {
            self.primary[self.primaryCount % self.mAvgInterval] = pval
        }
        self.primaryCount += 1
        self.mavgPrimary = self.primary.average() * 1000.0
    }
    
    // Calculate moving average intensity for ambientintensity.
    func movingAverageAmbient() {
        let aval = Float(self.lastLightEstimate!.ambientIntensity)/1000.0
        if self.ambientCount < self.mAvgInterval {
            self.ambient.append(aval)
        } else {
            self.ambient[self.ambientCount % self.mAvgInterval] = aval
        }
        self.ambientCount += 1
        self.mavgAmbient = self.ambient.average() * 1000.0
    }
    
    func printHarmonics(lightEstimate: ARDirectionalLightEstimate) {
        let spHarmonics = lightEstimate.sphericalHarmonicsCoefficients
        let floatSz = 4 // 4 bytes.
        let blockSz = 36 // 36 bytes for each channel.

        var redArr: [Float] = []
        for offset in stride(from: 0, to: blockSz, by: floatSz) {
            redArr.append(spHarmonics.withUnsafeBytes{ $0.load(fromByteOffset: offset, as: Float.self)})
        }

        self.harmonicPrint[0] += redArr[1]
        self.harmonicPrint[1] += redArr[2]
        self.harmonicPrint[2] += redArr[3]
        self.harmonicPrintCount += 1
        let COUNT_MAX = 20
        if self.harmonicPrintCount == COUNT_MAX {
            print ("Red coeffs: \(self.harmonicPrint[0]/Float(COUNT_MAX)), \(self.harmonicPrint[1]/Float(COUNT_MAX)), \(self.harmonicPrint[2]/Float(COUNT_MAX))")
            self.harmonicPrintCount = 0
            self.harmonicPrint = [0, 0, 0]
        }
    }

    @IBAction func onClick(_ sender: Any) {
        var lightDict: [String:[Float]]?
        var vertexNormals: [[Float]]?
        var vertices2D: [[Int]]?
        var pngData: Data?
        concurrentPhotoQueue.sync {
            if self.lastCapturedImage == nil {
                return
            }
            //self.pngData =  UIImage(pixelBuffer: lastImage)?.pngData()
            pngData = pixelBufferToUIImage().pngData()
            if pngData == nil {
                return
            }
            
            vertices2D = projectedVertices()
            if vertices2D == nil {
                return
            }
            
            vertexNormals = calcNormals()
            if vertexNormals == nil {
                return
            }
            
            lightDict = lightEstimate()
            if lightDict == nil {
                return
            }
        }
        
        guard let jsonVertices2D = try? JSONSerialization.data(withJSONObject: vertices2D!) else {
            return
        }
        
        guard let jsonLight = try? JSONSerialization.data(withJSONObject: lightDict!) else {
            return
        }
        
        guard let jsonVertexNormals =  try? JSONSerialization.data(withJSONObject: vertexNormals!) else {
            return
        }
        
        guard let jsonTriangleIdx =  try? JSONSerialization.data(withJSONObject: self.triangleIndices!) else {
            return
        }
        
        uploadData({ multipartFormData in
            multipartFormData.append(pngData!.base64EncodedData(), withName: "fileset", mimeType: "image/png")
            multipartFormData.append(jsonVertices2D, withName: "vertices")
            multipartFormData.append(jsonLight, withName: "lighting")
            multipartFormData.append(jsonTriangleIdx, withName: "triangleIndices")
            multipartFormData.append(jsonVertexNormals, withName: "vertexNormals")
        }, uploadURL)
    }
    
    // uploadData uploads given multipart form data to given URL.
    func uploadData(_ data: @escaping (Alamofire.MultipartFormData)->Void,  _ url: String) {
        Alamofire.upload(multipartFormData: data, to: url) { result in
            switch result {
            case .success(let upload, _, _):
                upload.uploadProgress{progress in
                    //print ("Upload progress \(progress.fractionCompleted)")
                    self.progressView.setProgress(Float(progress.fractionCompleted), animated: true)
                }
                
                upload.responseJSON { response in
                    //print ("Upload complete")
                    self.progressView.setProgress(0.0, animated: true)
                }
            case .failure(let encodingError):
                print (encodingError)
            }
        }
    }
    
    func pixelBufferToUIImage() -> UIImage {
        let ciImage = CIImage(cvPixelBuffer: self.lastCapturedImage!)
        let context = CIContext(options: nil)
        let cgImage = context.createCGImage(ciImage, from: ciImage.extent)
        let uiImage = UIImage(cgImage: cgImage!)
        return uiImage
    }
    
    func lightEstimate() -> [String:[Float]]? {
        guard let lightEstimate = self.lastLightEstimate else {
            return nil
        }
        let spHarmonics = lightEstimate.sphericalHarmonicsCoefficients
        var lightDict: [String:[Float]] = [:]
        lightDict["colorTemperature"] = [Float(lightEstimate.ambientColorTemperature)]
        lightDict["lumenIntensity"] = [self.mavgAmbient]
        lightDict["primaryIntensity"] = [self.mavgPrimary]
        lightDict["primaryDirection"] = self.mavgDir
        
        let floatSz = 4 // 4 bytes.
        let blockSz = 36 // 36 bytes for each channel.
        
        var redArr: [Float] = []
        for offset in stride(from: 0, to: blockSz, by: floatSz) {
            redArr.append(spHarmonics.withUnsafeBytes{ $0.load(fromByteOffset: offset, as: Float.self)})
        }
        lightDict["red"] = redArr
        
        var greenArr: [Float] = []
        for offset in stride(from: blockSz, to: 2*blockSz, by: floatSz) {
            greenArr.append(spHarmonics.withUnsafeBytes{ $0.load(fromByteOffset: offset, as: Float.self)})
        }
        lightDict["green"] = greenArr
        
        var blueArr: [Float] = []
        for offset in stride(from: 2*blockSz, to: 3*blockSz, by: floatSz) {
            blueArr.append(spHarmonics.withUnsafeBytes{ $0.load(fromByteOffset: offset, as: Float.self)})
        }
        lightDict["blue"] = blueArr
        return lightDict
    }
    
    func projectedVertices() -> [[Int]]? {
        guard let image = self.lastCapturedImage else {
            return nil
        }
        guard let faceAnchor = self.lastFaceAnchor else {
            return nil
        }
        guard let camera = self.lastCamera else {
            return nil
        }
        let geometry = faceAnchor.geometry
        let vertices = geometry.vertices
        
        let width = CVPixelBufferGetWidth(image)
        let height = CVPixelBufferGetHeight(image)
        
        let textureCoordinates = vertices.map { vertex -> [Int] in
            let world_vector3 = toWorldCords(vertex, faceAnchor)
            let pt = camera.projectPoint(world_vector3,
                orientation: .portrait,
                viewportSize: CGSize(
                    width: CGFloat(height),
                    height: CGFloat(width)))
            return [Int(pt.y), Int(pt.x)]
        }
        return textureCoordinates
    }
    
    // toWorldCords converts given face vertex from face
    // coordinate system to world coordinates.
    func toWorldCords(_ vertex: vector_float3, _ faceAnchor: ARFaceAnchor) -> simd_float3 {
        let vertex4 = vector_float4(vertex.x, vertex.y, vertex.z, 1)
        let world_vertex4 = simd_mul(faceAnchor.transform, vertex4)
        return simd_float3(x: world_vertex4.x, y: world_vertex4.y, z: world_vertex4.z)
    }
    
    // toFaceCords converts given vector from world coordinates
    // to face coordinates.
    func toFaceCords(_ vertex: vector_float3, _ faceAnchor: ARFaceAnchor) -> [Float] {
        let vertex4 = vector_float4(vertex.x, vertex.y, vertex.z, 1)
        let face_vertex4 = simd_mul(faceAnchor.transform.inverse, vertex4)
        return [face_vertex4.x, face_vertex4.y, face_vertex4.z]
    }
    
    func calcNormals() -> [[Float]]? {
        guard let faceAnchor = self.lastFaceAnchor else {
            return nil
        }
        
        //let vertices = faceAnchor.geometry.vertices
        self.triangleIndices = faceAnchor.geometry.triangleIndices
        guard let triangleIndices = self.triangleIndices else {
            return nil
        }
        let triangleCount = faceAnchor.geometry.triangleCount
        let vertices3D = faceAnchor.geometry.vertices
        
        var count = 0
        var vertexNormals = [simd_float3](repeating: simd_float3(0.0, 0.0, 0.0), count: vertices3D.count)
        while count != triangleCount {
            let v0 = vertices3D[Int(triangleIndices[count*3])]
            let v1 = vertices3D[Int(triangleIndices[count*3 + 1])]
            let v2 = vertices3D[Int(triangleIndices[count*3 + 2])]
            let cp = simd_cross(v1 - v2, v1 - v0)
            
            // Sum vertex normals with cross product.
            vertexNormals[Int(triangleIndices[count*3])] += cp
            vertexNormals[Int(triangleIndices[count*3 + 1])] += cp
            vertexNormals[Int(triangleIndices[count*3 + 2])] += cp
            
            count += 1
        }
        
        let vNormals = vertexNormals.map { vn -> [Float] in
            let sn = simd_normalize(vn)
            return [Float(sn.x), Float(sn.y), Float(sn.z)]
        }
        
        return vNormals
    }

    // prepare is run just before segue. It captures last stored image
    // and send the data to the image view controller.
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        concurrentPhotoQueue.sync {
            guard let lastImage = self.lastCapturedImage else {
                return
            }
            let destVC: ImageViewController = segue.destination as! ImageViewController
            destVC.capturedImage =  UIImage(pixelBuffer: lastImage)
        }
    }
}

// Extension of UIImage to convert CVPixelBuffer to UIImagge object.
extension UIImage {
    public convenience init?(pixelBuffer: CVPixelBuffer) {
        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)
        guard let cImage = cgImage else {
            return nil
        }
        self.init(cgImage: cImage)
    }
}

extension Sequence where Element: AdditiveArithmetic {
    /// Returns the total sum of all elements in the sequence
    func sum() -> Element { reduce(.zero, +) }
}

extension Collection where Element: BinaryFloatingPoint {
    /// Returns the average of all elements in the array
    func average() -> Element { isEmpty ? .zero : Element(sum()) / Element(count) }
}
