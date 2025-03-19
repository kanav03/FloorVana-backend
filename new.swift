import UIKit

class DrawViewController: UIViewController {

    @IBOutlet weak var canvasContainerView: UIView!
    @IBOutlet weak var inputDimensionsButton: UIButton!

    private var currentLine: CAShapeLayer?
    private var path = UIBezierPath()
    private var lastPoint: CGPoint?
    private var allLines: [(layer: CAShapeLayer, path: UIBezierPath, textLayer: CATextLayer?)] = []
    private let closeThreshold: CGFloat = 20
    private var startPoint: CGPoint?
    private var lengthTextLayer: CATextLayer?
    private var startPointLayer: CAShapeLayer?
    
    private var tutorialOverlay: UIView?
    private var handImageView: UIImageView?
    
    // Required image size for the synthesis model
    private let requiredSize: CGFloat = 256

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCanvasView()
        setupNavigationButtons()
        self.title = "Let's Plan"
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 21, weight: .bold)
        ]
        navigationController?.navigationBar.titleTextAttributes = attributes

        if isFirstLaunch() {
            showTutorialOverlay()
        }
        
        // Configure Generate button appearance
        inputDimensionsButton.layer.cornerRadius = 12
        inputDimensionsButton.backgroundColor = UIColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.85)
        inputDimensionsButton.setTitleColor(UIColor(red: 0.75, green: 0.6, blue: 0.25, alpha: 1.0), for: .normal)
        inputDimensionsButton.layer.borderWidth = 1
        inputDimensionsButton.layer.borderColor = UIColor(red: 0.75, green: 0.6, blue: 0.25, alpha: 1.0).cgColor
    }

    // MARK: - Tutorial Implementation
    private func isFirstLaunch() -> Bool {
        let launchedBefore = UserDefaults.standard.bool(forKey: "HasSeenTutorial")
        return !launchedBefore
    }
    
    private func showTutorialOverlay() {
        guard tutorialOverlay == nil else { return } // Prevent multiple overlays

        // Cover full screen including tab bar
        if let window = UIApplication.shared.windows.first {
            tutorialOverlay = UIView(frame: window.bounds)
            tutorialOverlay?.backgroundColor = UIColor.black.withAlphaComponent(0.85) // Darker background
            tutorialOverlay?.alpha = 0.0
            window.addSubview(tutorialOverlay!)
        }
        UIView.animate(withDuration: 0.3) {
            self.tutorialOverlay?.alpha = 1.0 // Smooth fade-in
        }

        // Move text further up
        let tutorialLabel = UILabel(frame: CGRect(x: 20, y: view.center.y - 250, width: view.bounds.width - 40, height: 60))
        tutorialLabel.text = "Draw the outlines of your dream"
        tutorialLabel.textColor = .white
        tutorialLabel.textAlignment = .center
        tutorialLabel.font = UIFont.boldSystemFont(ofSize: 22)
        tutorialLabel.numberOfLines = 2
        tutorialOverlay?.addSubview(tutorialLabel)

        // Hand icon (finger)
        handImageView = UIImageView(image: UIImage(systemName: "hand.point.up.fill"))
        handImageView?.tintColor = .white
        handImageView?.frame = CGRect(x: view.center.x - 35, y: view.center.y - 100, width: 50, height: 50)
        tutorialOverlay?.addSubview(handImageView!)
        
        // Line drawing layer
        let drawingLayer = CAShapeLayer()
        drawingLayer.strokeColor = UIColor.white.cgColor
        drawingLayer.lineWidth = 3
        drawingLayer.fillColor = nil
        drawingLayer.lineCap = .round
        tutorialOverlay?.layer.addSublayer(drawingLayer)

        animateHandDrawing(with: drawingLayer) // Hand moves & draws the square

        // Got It Button (Black Background with Matte Gold Text)
        let gotItButton = UIButton(frame: CGRect(x: view.center.x - 70, y: view.center.y + 180, width: 140, height: 50))
        gotItButton.setTitle("Got It", for: .normal)
        gotItButton.layer.borderWidth=1
        gotItButton.layer.borderColor=UIColor(red: 0.75, green: 0.6, blue: 0.25, alpha: 1.0).cgColor
        gotItButton.setTitleColor(UIColor(red: 0.75, green: 0.6, blue: 0.25, alpha: 1.0), for: .normal) // Matte Gold Text
        gotItButton.titleLabel?.font = UIFont.boldSystemFont(ofSize: 20)
        gotItButton.layer.cornerRadius = 12
        gotItButton.addTarget(self, action: #selector(dismissTutorial), for: .touchUpInside)
        tutorialOverlay?.addSubview(gotItButton)
    }

    private func animateHandDrawing(with drawingLayer: CAShapeLayer) {
        guard let hand = handImageView else { return }

        let startX = view.center.x - 80
        let startY = view.center.y - 80
        let squareSize: CGFloat = 160 // Increased square size

        let path = UIBezierPath()
        path.move(to: CGPoint(x: startX, y: startY))
        path.addLine(to: CGPoint(x: startX + squareSize, y: startY))
        path.addLine(to: CGPoint(x: startX + squareSize, y: startY + squareSize))
        path.addLine(to: CGPoint(x: startX, y: startY + squareSize))
        path.addLine(to: CGPoint(x: startX, y: startY)) // Close the square

        // Assign path to drawing layer
        drawingLayer.path = path.cgPath

        // Hand animation (syncing speed with line drawing)
        let animation = CAKeyframeAnimation(keyPath: "position")
        animation.path = path.cgPath
        animation.duration = 4.0 // Both hand and line move at the same speed
        animation.repeatCount = .infinity
        animation.timingFunction = CAMediaTimingFunction(name: .linear)

        hand.layer.add(animation, forKey: "handAnimation")

        // Line drawing animation (exact same timing as hand movement)
        let drawAnimation = CABasicAnimation(keyPath: "strokeEnd")
        drawAnimation.fromValue = 0
        drawAnimation.toValue = 1
        drawAnimation.duration = 4.0 // Matches hand movement
        drawAnimation.repeatCount = .infinity
        drawAnimation.timingFunction = CAMediaTimingFunction(name: .linear)

        drawingLayer.add(drawAnimation, forKey: "drawAnimation")
    }

    @objc private func dismissTutorial() {
        UIView.animate(withDuration: 0.3, animations: {
            self.tutorialOverlay?.alpha = 0.0 // Smooth fade-out
        }) { _ in
            self.tutorialOverlay?.removeFromSuperview()
            self.tutorialOverlay = nil
            self.handImageView?.layer.removeAllAnimations()
            UserDefaults.standard.set(true, forKey: "HasSeenTutorial")
        }
    }

    // MARK: - canvas
    private func setupCanvasView() {
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        canvasContainerView.addGestureRecognizer(panGesture)
        canvasContainerView.backgroundColor = UIColor(red: 0.91, green: 0.91, blue: 0.91, alpha: 0.95)
        canvasContainerView.layer.shadowColor = UIColor.black.cgColor
    }

    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        let touchPoint = gesture.location(in: canvasContainerView)

        switch gesture.state {
        case .began:
            if startPoint == nil {
                startPoint = touchPoint
                drawInitialPoint(at: touchPoint)
                lastPoint = touchPoint  // Start drawing from this point
            }

            path = UIBezierPath()
            path.move(to: lastPoint!)

            currentLine = CAShapeLayer()
            currentLine?.strokeColor = UIColor.black.cgColor
            currentLine?.lineWidth = 3
            currentLine?.fillColor = UIColor.clear.cgColor
            currentLine?.lineCap = .round

            canvasContainerView.layer.addSublayer(currentLine!)

            lengthTextLayer = CATextLayer()
            lengthTextLayer?.fontSize = 14
            lengthTextLayer?.foregroundColor = UIColor.gray.cgColor
            lengthTextLayer?.backgroundColor = UIColor.clear.cgColor
            lengthTextLayer?.contentsScale = UIScreen.main.scale
            canvasContainerView.layer.addSublayer(lengthTextLayer!)

        case .changed:
            let adjustedPoint = getStraightPoint(from: path.currentPoint, to: touchPoint)
            path.removeAllPoints()
            path.move(to: lastPoint!)
            path.addLine(to: adjustedPoint)
            currentLine?.path = path.cgPath

            updateLengthTextLayer(from: lastPoint!, to: adjustedPoint)

        case .ended:
            if let finalLine = currentLine {
                allLines.append((finalLine, path, lengthTextLayer))
            }

            if let start = startPoint, isCloseToStart(touchPoint, start: start) {
                closeShape()
            } else {
                lastPoint = path.currentPoint
            }

            currentLine = nil
            lengthTextLayer?.removeFromSuperlayer()

        default:
            break
        }
    }

    private func drawInitialPoint(at point: CGPoint) {
        startPointLayer = CAShapeLayer()
        startPointLayer?.path = UIBezierPath(ovalIn: CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10)).cgPath
        startPointLayer?.fillColor = UIColor.red.cgColor
        canvasContainerView.layer.addSublayer(startPointLayer!)
    }

    private func getStraightPoint(from start: CGPoint, to end: CGPoint) -> CGPoint {
        let dx = abs(end.x - start.x)
        let dy = abs(end.y - start.y)

        if dx > dy {
            return CGPoint(x: end.x, y: start.y)
        } else {
            return CGPoint(x: start.x, y: end.y)
        }
    }

    private func isCloseToStart(_ point: CGPoint, start: CGPoint) -> Bool {
        return abs(point.x - start.x) < closeThreshold && abs(point.y - start.y) < closeThreshold
    }

    private func closeShape() {
        guard let start = startPoint else { return }
        path.addLine(to: start)

        let closeLine = CAShapeLayer()
        closeLine.strokeColor = UIColor.black.cgColor
        closeLine.lineWidth = 3
        closeLine.fillColor = UIColor.clear.cgColor
        closeLine.path = path.cgPath

        allLines.append((closeLine, path, nil))
        canvasContainerView.layer.addSublayer(closeLine)
        lastPoint = nil
    }

    private func updateLengthTextLayer(from start: CGPoint, to end: CGPoint) {
        let distance = hypot(end.x - start.x, end.y - start.y)
        let lengthInCM = distance / 25  // More realistic cm scaling

        let midPoint = CGPoint(x: (start.x + end.x) / 2, y: (start.y + end.y) / 2)
        lengthTextLayer?.string = String(format: "%.1f cm", lengthInCM)
        lengthTextLayer?.frame = CGRect(x: midPoint.x - 30, y: midPoint.y - 12, width: 60, height: 24)
    }

    @objc private func undoLastLine() {
        guard let lastEntry = allLines.popLast() else { return }
        lastEntry.layer.removeFromSuperlayer()
        lastEntry.textLayer?.removeFromSuperlayer()
        lastPoint = allLines.last?.path.currentPoint ?? startPoint
    }

    @objc private func clearCanvas() {
        allLines.forEach {
            $0.layer.removeFromSuperlayer()
            $0.textLayer?.removeFromSuperlayer()
        }
        allLines.removeAll()
        lastPoint = nil

        startPointLayer?.removeFromSuperlayer()
        startPoint = nil
    }

    private func setupNavigationButtons() {
        let exportButton = UIBarButtonItem(image: UIImage(systemName: "square.and.arrow.up"), style: .plain, target: self, action: #selector(exportDrawing))
        let clearButton = UIBarButtonItem(image: UIImage(systemName: "trash"), style: .plain, target: self, action: #selector(clearCanvas))
        let undoButton = UIBarButtonItem(image: UIImage(systemName: "arrow.uturn.left"), style: .plain, target: self, action: #selector(undoLastLine))
        navigationItem.rightBarButtonItems = [exportButton, clearButton, undoButton]
    }
    
    // MARK: - Export Functionality
    @objc private func exportDrawing() {
        guard !allLines.isEmpty, startPoint != nil else {
            showAlert(title: "Nothing to Export", message: "Please draw something first.")
            return
        }
        
        // Get the processed image
        guard let processedImage = processDrawingForSynthesis() else {
            showAlert(title: "Error", message: "Failed to process the drawing.")
            return
        }
        
        // Create activity view controller
        let activityViewController = UIActivityViewController(
            activityItems: [processedImage],
            applicationActivities: nil
        )
        
        // Present the activity view controller
        if let popoverController = activityViewController.popoverPresentationController {
            popoverController.barButtonItem = navigationItem.rightBarButtonItems?.first
        }
        
        present(activityViewController, animated: true)
    }
    
    // MARK: - Get Drawing Paths
    private func getDrawingPath() -> UIBezierPath {
        let combinedPath = UIBezierPath()
        
        guard !allLines.isEmpty, let startPoint = startPoint else {
            return combinedPath
        }
        
        // Start with the first point
        combinedPath.move(to: startPoint)
        
        // Add all line segments
        for line in allLines {
            // Skip empty paths
            if line.path.isEmpty {
                continue
            }
            
            // Get the last point of the path
            let endPoint = line.path.currentPoint
            combinedPath.addLine(to: endPoint)
        }
        
        return combinedPath
    }
    
//     MARK: - Process Drawing Image for Synthesis Model
    private func processDrawingForSynthesis() -> UIImage? {
        let highResSize = CGSize(width: 1024, height: 1024)
        UIGraphicsBeginImageContextWithOptions(highResSize, false, 1.0) // false for transparent
        defer { UIGraphicsEndImageContext() }
        
        guard let context = UIGraphicsGetCurrentContext() else { return nil }
        
        // Set transparent background
        context.clear(CGRect(origin: .zero, size: highResSize))
        
        let combinedPath = getDrawingPath()
        guard !combinedPath.isEmpty else { return nil }
        
        if combinedPath.currentPoint != startPoint, let startPoint = startPoint {
            combinedPath.addLine(to: startPoint)
        }
        combinedPath.close()
        
        // Scale and center
        let margin: CGFloat = 40
        let boundingBox = combinedPath.bounds
        let scaleX = (highResSize.width - 2 * margin) / boundingBox.width
        let scaleY = (highResSize.height - 2 * margin) / boundingBox.height
        let scale = min(scaleX, scaleY)
        
        let centerX = (highResSize.width - boundingBox.width * scale) / 2 - boundingBox.minX * scale
        let centerY = (highResSize.height - boundingBox.height * scale) / 2 - boundingBox.minY * scale
        let transform = CGAffineTransform(translationX: centerX, y: centerY).scaledBy(x: scale, y: scale)
        
        let transformedPath = combinedPath.copy() as! UIBezierPath
        transformedPath.apply(transform)
        
        // Create a new image with the exact format needed by the synthesis model
        let width = Int(requiredSize)
        let height = Int(requiredSize)
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var rawData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        // Create a temporary context to draw the path
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let tempContext = CGContext(data: &rawData,
                                         width: width,
                                         height: height,
                                         bitsPerComponent: bitsPerComponent,
                                         bytesPerRow: bytesPerRow,
                                         space: colorSpace,
                                         bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        
        // Scale the path to fit the required size context
        let pathBounds = transformedPath.bounds
        let pathScale = min(
            CGFloat(width) / pathBounds.width,
            CGFloat(height) / pathBounds.height
        ) * 0.8 // 80% to leave some margin
        
        let pathCenterX = CGFloat(width) / 2 - (pathBounds.midX * pathScale)
        let pathCenterY = CGFloat(height) / 2 - (pathBounds.midY * pathScale)
        
        let finalTransform = CGAffineTransform(translationX: pathCenterX, y: pathCenterY)
            .scaledBy(x: pathScale, y: pathScale)
        
        let finalPath = transformedPath.copy() as! UIBezierPath
        finalPath.apply(finalTransform)
        
        // 1. First, fill with black
        tempContext.saveGState()
        tempContext.setFillColor(UIColor.black.cgColor)
        tempContext.addPath(finalPath.cgPath)
        tempContext.fillPath()
        tempContext.restoreGState()
        
        // 2. Create stroke path with thicker line
        let strokePath = finalPath.copy() as! UIBezierPath
        strokePath.lineWidth = 3.0  // Set appropriate line width for the boundary
        
        // 3. Draw the stroke in red (R=127)
        tempContext.saveGState()
        tempContext.setStrokeColor(UIColor(red: 127/255.0, green: 0, blue: 0, alpha: 1.0).cgColor)
        tempContext.addPath(strokePath.cgPath)
        tempContext.strokePath()
        tempContext.restoreGState()
        
        // 4. Set all fully transparent pixels to white with alpha=0
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * bytesPerRow) + (x * bytesPerPixel)
                
                // If alpha is 0, make it white with alpha=0
                if rawData[pixelIndex + 3] == 0 {
                    rawData[pixelIndex] = 255     // R
                    rawData[pixelIndex + 1] = 255 // G
                    rawData[pixelIndex + 2] = 255 // B
                    rawData[pixelIndex + 3] = 0   // A - transparent
                } else {
                    // Make all non-transparent pixels fully opaque
                    rawData[pixelIndex + 3] = 255
                }
            }
        }
        
        // Create the final image
        guard let finalContext = CGContext(data: &rawData,
                                         width: width,
                                         height: height,
                                         bitsPerComponent: bitsPerComponent,
                                         bytesPerRow: bytesPerRow,
                                         space: colorSpace,
                                         bitmapInfo: bitmapInfo.rawValue),
              let cgImage = finalContext.makeImage() else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }

    @IBAction func inputDimensionsTapped(_ sender: UIButton) {
        guard let synthesisReadyImage = processDrawingForSynthesis() else {
            showAlert(title: "Error", message: "Failed to process the drawing. Please draw a complete shape.")
            return
        }
        
        // Show a preview of the processed image
        showPreview(of: synthesisReadyImage)
        
        // Send the image directly to the server
        uploadImageToServer(synthesisReadyImage)
    }
    
    // Modify the uploadImageToServer method to preserve the image format
    private func uploadImageToServer(_ image: UIImage) {
        // First prepare the image for visualization
        guard let visualizationImage = prepareForVisualization(image) else {
            showAlert(title: "Error", message: "Failed to prepare image for visualization.")
            return
        }
        
        // Debug image format
        debugImageFormat(visualizationImage)
        
        // Get image data
        guard let imageData = visualizationImage.pngData() else {
            showAlert(title: "Error", message: "Failed to convert image to PNG data.")
            return
        }
        
        // Rest of your existing upload code...
        let url = URL(string: "https://w960g57g-5000.inc1.devtunnels.ms/generate_floorplan")! // Note: added endpoint
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Create form data with the visualization-ready image
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"floorplan.png\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        // Show loading indicator
        let loadingAlert = UIAlertController(title: "Processing", message: "Generating floor plan...", preferredStyle: .alert)
        present(loadingAlert, animated: true)
        
        // Send request
        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                // Dismiss loading indicator
                loadingAlert.dismiss(animated: true) {
                    // Check for network error
                    if let error = error {
                        self?.showAlert(title: "Network Error", message: "Failed to connect to server: \(error.localizedDescription)")
                        return
                    }
                    
                    // Check HTTP response
                    if let httpResponse = response as? HTTPURLResponse {
                        print("HTTP Response Status: \(httpResponse.statusCode)")
                        
                        if httpResponse.statusCode == 404 {
                            self?.showAlert(title: "Server Error", message: "Endpoint not found (404). Please check the server URL and ensure the Flask server is running.")
                            return
                        }
                        
                        if httpResponse.statusCode != 200 {
                            self?.showAlert(title: "Server Error", message: "Server returned status code \(httpResponse.statusCode)")
                            return
                        }
                    }
                    
                    guard let data = data else {
                        self?.showAlert(title: "Error", message: "No data received from server")
                        return
                    }
                    
                    // Process response
                    do {
                        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                            // Check for error in response
                            if let errorMessage = json["error"] as? String {
                                self?.showAlert(title: "Server Error", message: errorMessage)
                                return
                            }
                            
                            // Extract normalized image
                            if let images = json["images"] as? [String: Any],
                               let normalizedBase64 = images["normalized"] as? String,
                               let normalizedData = Data(base64Encoded: normalizedBase64) {
                                
                                // Verify image hash matches original
                                let returnedHash = self?.calculateImageHash(data: normalizedData) ?? ""
                                print("Returned image hash: \(returnedHash)")
                                print("Hash match: \(originalHash == returnedHash)")
                                
                                // Show the normalized image
                                if let normalizedImage = UIImage(data: normalizedData) {
                                    self?.showResultImage(normalizedImage)
                                    
                                    // Also show visualization if available
                                    if let vizBase64 = images["visualization"] as? String,
                                       let vizData = Data(base64Encoded: vizBase64),
                                       let vizImage = UIImage(data: vizData) {
                                        self?.showVisualizationImage(vizImage)
                                    }
                                } else {
                                    self?.showAlert(title: "Error", message: "Failed to decode normalized image")
                                }
                            } else {
                                self?.showAlert(title: "Error", message: "Normalized image not found in response")
                            }
                        } else {
                            self?.showAlert(title: "Error", message: "Invalid JSON response from server")
                        }
                    } catch {
                        self?.showAlert(title: "Error", message: "Failed to parse response: \(error.localizedDescription)")
                    }
                }
            }
        }
        
        task.resume()
    }

    // Add this helper method to calculate image hash for verification
    private func calculateImageHash(data: Data) -> String {
        var hasher = Hasher()
        hasher.combine(data)
        return "\(hasher.finalize())"
    }
    private func showResultImage(_ image: UIImage) {
        let resultVC = UIViewController()
        resultVC.view.backgroundColor = .white
        
        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        
        let closeButton = UIButton(type: .system)
        closeButton.setTitle("Close", for: .normal)
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        closeButton.addTarget(resultVC, action: #selector(UIViewController.dismiss(animated:completion:)), for: .touchUpInside)
        
        resultVC.view.addSubview(imageView)
        resultVC.view.addSubview(closeButton)
        
        NSLayoutConstraint.activate([
            imageView.centerXAnchor.constraint(equalTo: resultVC.view.centerXAnchor),
            imageView.centerYAnchor.constraint(equalTo: resultVC.view.centerYAnchor),
            imageView.widthAnchor.constraint(equalToConstant: 300),
            imageView.heightAnchor.constraint(equalToConstant: 300),
            
            closeButton.centerXAnchor.constraint(equalTo: resultVC.view.centerXAnchor),
            closeButton.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 20),
            closeButton.heightAnchor.constraint(equalToConstant: 44)
        ])
        
        resultVC.modalPresentationStyle = .fullScreen
        present(resultVC, animated: true)
    }
    
    private func showPreview(of image: UIImage) {
        let previewVC = ImagePreviewViewController()
        previewVC.image = image
        previewVC.modalPresentationStyle = .pageSheet
        present(previewVC, animated: true)
    }
    
    private func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }

    @objc private func imageSaved(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            print("Error saving image: \(error.localizedDescription)")
            showAlert(title: "Error", message: "Failed to save image: \(error.localizedDescription)")
        } else {
            print("Image saved successfully")
        }
    }
    
    // MARK: - Additional Export Options
    private func showExportOptions() {
        guard let imageToExport = processDrawingForSynthesis() else {
            showAlert(title: "Error", message: "Failed to process the drawing")
            return
        }
        
        let actionSheet = UIAlertController(title: "Export Options", message: "Choose export format", preferredStyle: .actionSheet)
        
        // PNG Export
        actionSheet.addAction(UIAlertAction(title: "Export as PNG", style: .default) { [weak self] _ in
            self?.exportAs(image: imageToExport, fileType: "png")
        })
        
        // JPEG Export
        actionSheet.addAction(UIAlertAction(title: "Export as JPEG", style: .default) { [weak self] _ in
            self?.exportAs(image: imageToExport, fileType: "jpg")
        })
        
        // Save to Photos
        actionSheet.addAction(UIAlertAction(title: "Save to Photos", style: .default) { [weak self] _ in
            UIImageWriteToSavedPhotosAlbum(imageToExport, self, #selector(self?.imageSaved(_:didFinishSavingWithError:contextInfo:)), nil)
        })
        
        // Cancel option
        actionSheet.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        
        // For iPad support
        if let popoverController = actionSheet.popoverPresentationController {
            popoverController.barButtonItem = navigationItem.rightBarButtonItems?.first
        }
        
        present(actionSheet, animated: true)
    }
    
    private func exportAs(image: UIImage, fileType: String) {
        var data: Data?
        var mimeType: String
        
        if fileType == "png" {
            data = image.pngData()
            mimeType = "image/png"
        } else {
            data = image.jpegData(compressionQuality: 0.9)
            mimeType = "image/jpeg"
        }
        
        guard let imageData = data else {
            showAlert(title: "Error", message: "Failed to create image data")
            return
        }
        
        // Create a temporary file
        let temporaryDirectoryURL = FileManager.default.temporaryDirectory
        let temporaryFileURL = temporaryDirectoryURL.appendingPathComponent("export_drawing.\(fileType)")
        
        do {
            try imageData.write(to: temporaryFileURL)
            
            // Show share sheet
            let activityViewController = UIActivityViewController(
                activityItems: [temporaryFileURL],
                applicationActivities: nil
            )
            
            // For iPad support
            if let popoverController = activityViewController.popoverPresentationController {
                popoverController.barButtonItem = navigationItem.rightBarButtonItems?.first
            }
            
            present(activityViewController, animated: true)
        } catch {
            showAlert(title: "Error", message: "Failed to export: \(error.localizedDescription)")
        }
    }

    private func debugImageChannels(_ image: UIImage) -> [UIImage] {
        guard let cgImage = image.cgImage else { return [] }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var rawData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: &rawData,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: bitsPerComponent,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo.rawValue) else {
            return []
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var channelImages: [UIImage] = []
        
        // Create separate images for each channel
        for channel in 0..<4 {
            var channelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
            
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = (y * width + x) * bytesPerPixel
                    let value = rawData[pixelIndex + channel]
                    
                    // Set all RGB channels to the value of the current channel
                    channelData[pixelIndex] = value     // R
                    channelData[pixelIndex + 1] = value // G
                    channelData[pixelIndex + 2] = value // B
                    channelData[pixelIndex + 3] = 255   // A
                }
            }
            
            guard let channelContext = CGContext(data: &channelData,
                                               width: width,
                                               height: height,
                                               bitsPerComponent: bitsPerComponent,
                                               bytesPerRow: bytesPerRow,
                                               space: colorSpace,
                                               bitmapInfo: bitmapInfo.rawValue),
                  let channelCGImage = channelContext.makeImage() else {
                continue
            }
            
            channelImages.append(UIImage(cgImage: channelCGImage))
        }
        
        return channelImages
    }

    // Add a method to show the visualization image
    private func showVisualizationImage(_ image: UIImage) {
        let resultVC = UIViewController()
        resultVC.view.backgroundColor = .white
        
        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        
        let titleLabel = UILabel()
        titleLabel.text = "Visualization Result"
        titleLabel.font = UIFont.boldSystemFont(ofSize: 18)
        titleLabel.textAlignment = .center
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        
        let closeButton = UIButton(type: .system)
        closeButton.setTitle("Close", for: .normal)
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        closeButton.addTarget(resultVC, action: #selector(UIViewController.dismiss(animated:completion:)), for: .touchUpInside)
        
        resultVC.view.addSubview(titleLabel)
        resultVC.view.addSubview(imageView)
        resultVC.view.addSubview(closeButton)
        
        NSLayoutConstraint.activate([
            titleLabel.topAnchor.constraint(equalTo: resultVC.view.safeAreaLayoutGuide.topAnchor, constant: 20),
            titleLabel.leadingAnchor.constraint(equalTo: resultVC.view.leadingAnchor, constant: 20),
            titleLabel.trailingAnchor.constraint(equalTo: resultVC.view.trailingAnchor, constant: -20),
            
            imageView.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 20),
            imageView.centerXAnchor.constraint(equalTo: resultVC.view.centerXAnchor),
            imageView.widthAnchor.constraint(equalToConstant: 300),
            imageView.heightAnchor.constraint(equalToConstant: 300),
            
            closeButton.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 20),
            closeButton.centerXAnchor.constraint(equalTo: resultVC.view.centerXAnchor),
            closeButton.heightAnchor.constraint(equalToConstant: 44)
        ])
        
        resultVC.modalPresentationStyle = .fullScreen
        present(resultVC, animated: true)
    }

    private func debugImageFormat(_ image: UIImage) {
        guard let cgImage = image.cgImage else { return }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var rawData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: &rawData,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: bitsPerComponent,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo.rawValue) else {
            return
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Sample a few pixels to check format
        let samplePoints = [
            (width/4, height/4),
            (width/2, height/2),
            (3*width/4, 3*height/4)
        ]
        
        print("Image format debug:")
        print("Dimensions: \(width)x\(height)")
        print("Sample pixels:")
        
        for (x, y) in samplePoints {
            let pixelIndex = (y * width + x) * bytesPerPixel
            let r = rawData[pixelIndex]
            let g = rawData[pixelIndex + 1]
            let b = rawData[pixelIndex + 2]
            let a = rawData[pixelIndex + 3]
            print("Pixel at (\(x), \(y)): R=\(r), G=\(g), B=\(b), A=\(a)")
        }
    }
    
    private func prepareForVisualization(_ normalizedImage: UIImage) -> UIImage? {
        // Create a new image with exact specifications required by deeplayout.cpp
        let size = CGSize(width: 256, height: 256)
        let bytesPerRow = Int(size.width) * 4
        var rawData = [UInt8](repeating: 0, count: Int(size.width * size.height * 4))
        
        guard let cgImage = normalizedImage.cgImage,
              let context = CGContext(data: &rawData,
                                    width: Int(size.width),
                                    height: Int(size.height),
                                    bitsPerComponent: 8,
                                    bytesPerRow: bytesPerRow,
                                    space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return nil
        }
        
        // Draw original normalized image
        context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        
        // Process pixels to exactly match deeplayout.cpp ReadImageData requirements:
        // - boundary_map (Red): 127 for walls, 255 for front door
        // - interior_wall_map (Green): 127 for interior walls
        // - label_map (Blue): 100+ for room labels
        // - inside_map (Alpha): 255 for inside, 0 for outside
        for y in 0..<Int(size.height) {
            for x in 0..<Int(size.width) {
                let pixelIndex = (y * bytesPerRow) + (x * 4)
                
                let r = rawData[pixelIndex]
                let g = rawData[pixelIndex + 1]
                let a = rawData[pixelIndex + 3]
                
                if a > 0 { // Inside area
                    if r > 100 && r < 150 { // Regular wall
                        rawData[pixelIndex] = 127     // R - Exact wall value
                        rawData[pixelIndex + 1] = 0   // G
                        rawData[pixelIndex + 2] = 0   // B
                        rawData[pixelIndex + 3] = 255 // A
                    } else if r > 200 { // Front door
                        rawData[pixelIndex] = 255     // R - Front door
                        rawData[pixelIndex + 1] = 0   // G
                        rawData[pixelIndex + 2] = 0   // B
                        rawData[pixelIndex + 3] = 255 // A
                    } else if g > 100 { // Interior wall
                        rawData[pixelIndex] = 0       // R
                        rawData[pixelIndex + 1] = 127 // G - Interior wall
                        rawData[pixelIndex + 2] = 0   // B
                        rawData[pixelIndex + 3] = 255 // A
                    } else { // Room area
                        rawData[pixelIndex] = 0       // R
                        rawData[pixelIndex + 1] = 0   // G
                        rawData[pixelIndex + 2] = 100 // B - Room label
                        rawData[pixelIndex + 3] = 255 // A
                    }
                } else { // Outside area
                    rawData[pixelIndex] = 0     // R
                    rawData[pixelIndex + 1] = 0 // G
                    rawData[pixelIndex + 2] = 0 // B
                    rawData[pixelIndex + 3] = 0 // A
                }
            }
        }
        
        // Add a small delay to ensure pixel operations complete
        Thread.sleep(forTimeInterval: 0.1)
        
        // Create final image
        guard let finalContext = CGContext(data: &rawData,
                                         width: Int(size.width),
                                         height: Int(size.height),
                                         bitsPerComponent: 8,
                                         bytesPerRow: bytesPerRow,
                                         space: CGColorSpaceCreateDeviceRGB(),
                                         bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let finalImage = finalContext.makeImage() else {
            return nil
        }
        
        return UIImage(cgImage: finalImage)
    }
}

// Simple preview controller for the processed image
class ImagePreviewViewController: UIViewController {
    var image: UIImage?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        view.backgroundColor = .white
        
        // Create close button
        let closeButton = UIButton(type: .system)
        closeButton.setTitle("Close", for: .normal)
        closeButton.addTarget(self, action: #selector(dismissPreview), for: .touchUpInside)
        
        // Create image view
        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFit
        imageView.clipsToBounds = true
        
        // Title label
        let titleLabel = UILabel()
        titleLabel.text = "Processed Floor Plan Image"
        titleLabel.font = UIFont.boldSystemFont(ofSize: 18)
        titleLabel.textAlignment = .center
        
        // Description label
        let descLabel = UILabel()
        descLabel.text = "This image has been processed to 256×256 pixels with the correct RGBA format for the synthesis model."
        descLabel.font = UIFont.systemFont(ofSize: 14)
        descLabel.textAlignment = .center
        descLabel.numberOfLines = 0
        
        // Add to subviews with Auto Layout
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        imageView.translatesAutoresizingMaskIntoConstraints = false
        descLabel.translatesAutoresizingMaskIntoConstraints = false
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(titleLabel)
        view.addSubview(imageView)
        view.addSubview(descLabel)
        view.addSubview(closeButton)
        
        NSLayoutConstraint.activate([
            titleLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            titleLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            titleLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            imageView.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 20),
            imageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            imageView.widthAnchor.constraint(equalToConstant: 300),
            imageView.heightAnchor.constraint(equalToConstant: 300),
            
            descLabel.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 20),
            descLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            descLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            closeButton.topAnchor.constraint(equalTo: descLabel.bottomAnchor, constant: 30),
            closeButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            closeButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    @objc private func dismissPreview() {
        dismiss(animated: true)
    }
}
