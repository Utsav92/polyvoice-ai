// Renders text to a transparent PNG using Core Text (correct shaping for Devanagari, CJK, etc.).
// usage: rendertext out.png "text" size weight(ultralight..black) hexcolor width align(left|center|right) lineHeightMultiple tracking
import AppKit
let a = CommandLine.arguments
guard a.count >= 9 else { print("usage"); exit(1) }
let out = a[1], text = a[2], size = CGFloat(Double(a[3])!), weightName = a[4], hex = a[5], width = CGFloat(Double(a[6])!), align = a[7], lh = CGFloat(Double(a[8])!)
let tracking = a.count > 9 ? CGFloat(Double(a[9])!) : 0
let weights: [String: NSFont.Weight] = ["ultralight": .ultraLight, "thin": .thin, "light": .light, "regular": .regular, "medium": .medium, "semibold": .semibold, "bold": .bold, "heavy": .heavy, "black": .black]
let font = NSFont.systemFont(ofSize: size, weight: weights[weightName] ?? .regular)
let v = UInt32(hex, radix: 16)!
let color = NSColor(srgbRed: CGFloat((v >> 16) & 255) / 255, green: CGFloat((v >> 8) & 255) / 255, blue: CGFloat(v & 255) / 255, alpha: 1)
let ps = NSMutableParagraphStyle(); ps.lineHeightMultiple = lh; ps.lineBreakMode = .byWordWrapping
ps.alignment = align == "center" ? .center : align == "right" ? .right : .left
let attr = NSAttributedString(string: text, attributes: [.font: font, .foregroundColor: color, .paragraphStyle: ps, .kern: tracking * size])
let opts: NSString.DrawingOptions = [.usesLineFragmentOrigin, .usesFontLeading]
let r = attr.boundingRect(with: NSSize(width: width, height: 10000), options: opts)
let pad: CGFloat = 8
let W = Int(ceil(max(r.width, 1)) + pad * 2), H = Int(ceil(r.height) + pad * 2)
let scale: CGFloat = 1
guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: W, pixelsHigh: H, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else { exit(2) }
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
attr.draw(with: NSRect(x: pad, y: pad, width: CGFloat(W) - pad * 2, height: CGFloat(H) - pad * 2), options: opts)
NSGraphicsContext.restoreGraphicsState()
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("\(W) \(H)")
