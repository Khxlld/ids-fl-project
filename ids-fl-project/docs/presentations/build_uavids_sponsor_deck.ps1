param(
    [string]$OutputPath = (Join-Path $PSScriptRoot "UAVIDS_Quantum_Secure_FL_Sponsor_Deck.pptx")
)

$ErrorActionPreference = "Stop"

function RGB([int]$r, [int]$g, [int]$b) {
    return $r + (256 * $g) + (65536 * $b)
}

$C = @{
    Navy       = RGB 8 20 34
    Navy2      = RGB 13 30 49
    Panel      = RGB 20 40 61
    Panel2     = RGB 25 48 71
    Border     = RGB 47 73 98
    White      = RGB 248 250 252
    Muted      = RGB 164 180 199
    Muted2     = RGB 119 140 163
    Teal       = RGB 16 197 139
    TealDark   = RGB 8 137 105
    Blue       = RGB 59 130 246
    Amber      = RGB 245 158 11
    Red        = RGB 248 113 113
    Green      = RGB 52 211 153
    Track      = RGB 34 54 75
}

$FONT = "Aptos"
$FONT_DISPLAY = "Aptos Display"

function Add-Box {
    param($Slide, [double]$Left, [double]$Top, [double]$Width, [double]$Height,
          [int]$Fill, [int]$Line, [double]$LineWeight = 1,
          [double]$Transparency = 0, [switch]$Square)
    $shapeType = if ($Square) { 1 } else { 5 }
    $shape = $Slide.Shapes.AddShape($shapeType, $Left, $Top, $Width, $Height)
    $shape.Fill.Visible = -1
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $Fill
    $shape.Fill.Transparency = $Transparency
    if ($LineWeight -le 0) {
        $shape.Line.Visible = 0
    } else {
        $shape.Line.Visible = -1
        $shape.Line.ForeColor.RGB = $Line
        $shape.Line.Weight = $LineWeight
    }
    return $shape
}

function Add-Circle {
    param($Slide, [double]$Left, [double]$Top, [double]$Size,
          [int]$Fill, [int]$Line, [double]$LineWeight = 1,
          [double]$Transparency = 0)
    $shape = $Slide.Shapes.AddShape(9, $Left, $Top, $Size, $Size)
    $shape.Fill.Visible = -1
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $Fill
    $shape.Fill.Transparency = $Transparency
    if ($LineWeight -le 0) {
        $shape.Line.Visible = 0
    } else {
        $shape.Line.Visible = -1
        $shape.Line.ForeColor.RGB = $Line
        $shape.Line.Weight = $LineWeight
    }
    return $shape
}

function Add-Text {
    param($Slide, [string]$Text, [double]$Left, [double]$Top,
          [double]$Width, [double]$Height, [double]$Size,
          [int]$Color, [switch]$Bold, [int]$Align = 1,
          [int]$VAlign = 1, [string]$Font = $FONT)
    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame2.VerticalAnchor = $VAlign
    $range = $shape.TextFrame.TextRange
    $range.Text = $Text
    $range.Font.Name = $Font
    $range.Font.Size = $Size
    $range.Font.Bold = if ($Bold) { -1 } else { 0 }
    $range.Font.Color.RGB = $Color
    $range.ParagraphFormat.Alignment = $Align
    return $shape
}

function Add-Line {
    param($Slide, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2,
          [int]$Color, [double]$Weight = 1.5, [double]$Transparency = 0,
          [switch]$ArrowEnd, [switch]$Dash)
    $line = $Slide.Shapes.AddLine($X1, $Y1, $X2, $Y2)
    $line.Line.ForeColor.RGB = $Color
    $line.Line.Weight = $Weight
    $line.Line.Transparency = $Transparency
    if ($ArrowEnd) { $line.Line.EndArrowheadStyle = 3 }
    if ($Dash) { $line.Line.DashStyle = 4 }
    return $line
}

function Add-Pill {
    param($Slide, [string]$Text, [double]$Left, [double]$Top,
          [double]$Width, [double]$Height, [int]$Fill, [int]$TextColor,
          [int]$Line = $Fill, [double]$FontSize = 10)
    [void](Add-Box $Slide $Left $Top $Width $Height $Fill $Line 0 0)
    [void](Add-Text $Slide $Text $Left $Top $Width $Height $FontSize $TextColor -Bold -Align 2 -VAlign 3)
}

function Add-Header {
    param($Slide, [string]$Kicker, [string]$Title, [string]$Number)
    [void](Add-Text $Slide $Kicker.ToUpperInvariant() 44 24 350 18 10 $C.Teal -Bold)
    [void](Add-Text $Slide $Title 44 45 780 42 28 $C.White -Bold -Font $FONT_DISPLAY)
    [void](Add-Text $Slide $Number 866 28 50 20 10 $C.Muted2 -Bold -Align 3)
    [void](Add-Line $Slide 44 94 916 94 $C.Border 1 0.2)
    [void](Add-Text $Slide "KAUST Academy  |  Cybersecurity Specialization" 44 516 420 12 8.5 $C.Muted2)
}

function Add-NumberedPoint {
    param($Slide, [string]$Number, [string]$Title, [string]$Body,
          [double]$Top, [int]$Accent)
    [void](Add-Circle $Slide 50 $Top 28 $Accent $Accent 0)
    [void](Add-Text $Slide $Number 50 $Top 28 28 11 $C.Navy -Bold -Align 2 -VAlign 3)
    [void](Add-Text $Slide $Title 92 ($Top - 1) 410 21 16 $C.White -Bold)
    [void](Add-Text $Slide $Body 92 ($Top + 22) 420 36 11.5 $C.Muted)
}

function Add-ServerIcon {
    param($Slide, [double]$Left, [double]$Top, [double]$Scale, [int]$Color)
    foreach ($offset in @(0, 14, 28)) {
        [void](Add-Box $Slide $Left ($Top + ($offset * $Scale)) (44 * $Scale) (10 * $Scale) $C.Navy $Color 1 0)
        [void](Add-Circle $Slide ($Left + (5 * $Scale)) ($Top + (($offset + 3) * $Scale)) (4 * $Scale) $Color $Color 0)
    }
}

function Add-DroneIcon {
    param($Slide, [double]$CenterX, [double]$CenterY, [double]$Scale, [int]$Color)
    [void](Add-Line $Slide ($CenterX - 14*$Scale) ($CenterY - 8*$Scale) ($CenterX + 14*$Scale) ($CenterY + 8*$Scale) $Color (1.5*$Scale))
    [void](Add-Line $Slide ($CenterX - 14*$Scale) ($CenterY + 8*$Scale) ($CenterX + 14*$Scale) ($CenterY - 8*$Scale) $Color (1.5*$Scale))
    [void](Add-Box $Slide ($CenterX - 8*$Scale) ($CenterY - 4*$Scale) (16*$Scale) (8*$Scale) $Color $Color 0 0)
    foreach ($xy in @(@(-17,-11),@(13,-11),@(-17,7),@(13,7))) {
        [void](Add-Circle $Slide ($CenterX + $xy[0]*$Scale) ($CenterY + $xy[1]*$Scale) (8*$Scale) $C.Navy $Color (1.2*$Scale))
    }
}

function Add-Notes {
    param($Slide, [string]$Text)
    try {
        foreach ($shape in $Slide.NotesPage.Shapes) {
            try {
                if ($shape.PlaceholderFormat.Type -eq 2) {
                    $shape.TextFrame.TextRange.Text = $Text
                    return
                }
            } catch {}
        }
    } catch {}
}

$sourcePath = Join-Path $PSScriptRoot "IDS.pptx"
if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Source deck not found: $sourcePath"
}

$outputFull = [System.IO.Path]::GetFullPath($OutputPath)
if ($outputFull -eq [System.IO.Path]::GetFullPath($sourcePath)) {
    throw "Output must not overwrite the teammate's source deck."
}

Copy-Item -LiteralPath $sourcePath -Destination $outputFull -Force

$ppt = New-Object -ComObject PowerPoint.Application
$presentation = $null
try {
    $presentation = $ppt.Presentations.Open($outputFull, $false, $false, $false)

    # Keep the original cover's embedded official logos, discard all other content.
    for ($i = $presentation.Slides.Count; $i -ge 2; $i--) {
        $presentation.Slides.Item($i).Delete()
    }
    $cover = $presentation.Slides.Item(1)
    for ($i = $cover.Shapes.Count; $i -ge 1; $i--) {
        $shape = $cover.Shapes.Item($i)
        if ($shape.Name -notin @("Picture 6", "Picture 7", "Picture 8", "Picture 9")) {
            $shape.Delete()
        }
    }
    $cover.FollowMasterBackground = 0
    $cover.Background.Fill.Solid()
    $cover.Background.Fill.ForeColor.RGB = $C.Navy

    # Cover: subtle network field.
    foreach ($ring in @(310, 250, 190)) {
        $r = Add-Circle $cover (678 - $ring/2) (230 - $ring/2) $ring $C.Navy $C.Blue 1 0.92
        $r.Fill.Visible = 0
    }
    $nodePositions = @(
        @(670,102), @(824,154), @(844,303), @(692,350), @(576,236)
    )
    foreach ($pos in $nodePositions) {
        [void](Add-Line $cover ($pos[0]+15) ($pos[1]+15) 757 240 $C.Blue 1.2 0.58)
        [void](Add-Circle $cover $pos[0] $pos[1] 30 $C.Navy2 $C.Teal 1.6 0)
        Add-DroneIcon $cover ($pos[0]+15) ($pos[1]+15) 0.42 $C.Teal
    }
    [void](Add-Circle $cover 707 190 100 $C.Navy2 $C.Amber 2 0)
    Add-ServerIcon $cover 735 211 0.72 $C.White
    Add-Pill $cover "SECURE FEDAVG" 711 264 92 18 $C.Amber $C.Navy $C.Amber 8.5

    [void](Add-Pill $cover "CYBERSECURITY PROGRESS PRESENTATION" 56 48 264 24 $C.Panel2 $C.Teal $C.Border 9)
    [void](Add-Text $cover "QUANTUM-SECURE" 56 96 540 36 17 $C.Teal -Bold)
    [void](Add-Text $cover "Federated Intrusion Detection" 56 132 570 57 34 $C.White -Bold -Font $FONT_DISPLAY)
    [void](Add-Text $cover "for UAV Networks" 56 188 520 46 31 $C.White -Bold -Font $FONT_DISPLAY)
    [void](Add-Text $cover "Collaborative threat detection without centralizing raw training data" 58 252 520 42 15 $C.Muted)
    [void](Add-Line $cover 58 316 560 316 $C.Border 1 0)
    [void](Add-Text $cover "Khalid Alkhaldi  |  Anas Alshehri  |  Feras Aloufi`rMeshal Alshawi  |  Ammar Alkayyal" 58 333 520 42 11.5 $C.White)
    [void](Add-Text $cover "Supervised by Dr. Muhammad Shahbaz Khan" 58 389 460 18 10.5 $C.Muted)

    # Reposition the four official logo images retained from the source deck.
    $logoLayout = @{
        "Picture 6" = @(64, 445, 72)
        "Picture 7" = @(185, 447, 125)
        "Picture 8" = @(365, 447, 144)
        "Picture 9" = @(575, 456, 164)
    }
    foreach ($name in $logoLayout.Keys) {
        $logo = $cover.Shapes.Item($name)
        $logo.LockAspectRatio = -1
        $logo.Left = $logoLayout[$name][0]
        $logo.Top = $logoLayout[$name][1]
        $logo.Width = $logoLayout[$name][2]
    }
    Add-Notes $cover @"
Timing: 20 seconds.
Opening: UAV nodes see different traffic, but they still need to learn from one another. Our project asks whether they can collaborate without uploading their raw training records - and whether the model exchange itself can be protected against tampering, impersonation, replay, and future quantum-capable attackers.
"@

    # Create five blank slides.
    $slides = @()
    for ($i = 2; $i -le 6; $i++) {
        $slide = $presentation.Slides.Add($i, 12)
        $slide.FollowMasterBackground = 0
        $slide.Background.Fill.Solid()
        $slide.Background.Fill.ForeColor.RGB = $C.Navy
        $slides += $slide
    }

    # Slide 2: problem.
    $s = $slides[0]
    Add-Header $s "The challenge" "UAV fleets need shared intelligence - without centralized logs" "01"
    Add-NumberedPoint $s "1" "Uneven local knowledge" "Each node can observe a different mix of normal and malicious behaviour." 130 $C.Teal
    Add-NumberedPoint $s "2" "Centralization creates exposure" "Uploading all records increases data concentration and dependence on one hub." 216 $C.Blue
    Add-NumberedPoint $s "3" "The learning channel is a target" "Updates can be copied, altered, replayed, or submitted under a false identity." 302 $C.Amber

    # Centralized-risk visual.
    [void](Add-Text $s "THE CENTRALIZATION TRADE-OFF" 584 126 310 18 9.5 $C.Muted2 -Bold -Align 2)
    $clients = @(@(585,175),@(585,255),@(585,335))
    foreach ($p in $clients) {
        [void](Add-Circle $s $p[0] $p[1] 45 $C.Navy2 $C.Teal 1.4)
        Add-DroneIcon $s ($p[0]+22.5) ($p[1]+22.5) 0.58 $C.Teal
        [void](Add-Line $s ($p[0]+52) ($p[1]+22) 765 270 $C.Red 1.6 0 -ArrowEnd)
    }
    [void](Add-Box $s 770 205 130 132 $C.Panel $C.Red 1.7)
    Add-ServerIcon $s 813 230 0.78 $C.White
    [void](Add-Text $s "RAW LOG HUB" 785 292 100 22 11 $C.White -Bold -Align 2)
    Add-Pill $s "privacy exposure" 686 355 112 22 $C.Panel2 $C.Red $C.Red 8.8
    Add-Pill $s "bottleneck" 807 355 88 22 $C.Panel2 $C.Red $C.Red 8.8

    [void](Add-Box $s 44 432 872 58 $C.Panel $C.Border 1 0)
    [void](Add-Text $s "122,171" 66 443 98 26 20 $C.Teal -Bold)
    [void](Add-Text $s "verified UAV network flows" 165 447 200 20 11 $C.White -Bold)
    [void](Add-Line $s 390 443 390 478 $C.Border 1)
    [void](Add-Text $s "NORMAL + 4 ATTACK FAMILIES" 418 447 220 18 10 $C.White -Bold)
    [void](Add-Text $s "Blackhole | Flooding | Sybil | Wormhole" 418 466 250 14 9 $C.Muted)
    [void](Add-Line $s 685 443 685 478 $C.Border 1)
    [void](Add-Text $s "BINARY GOAL" 712 445 90 15 9 $C.Muted2 -Bold)
    [void](Add-Text $s "Normal or Attack" 712 463 160 16 11 $C.White -Bold)
    Add-Notes $s @"
Timing: 60 seconds.
The key tension is collaboration versus concentration. Local-only learning can be inconsistent because the five source partitions have very different attack prevalence. Central pooling is an accuracy reference, but it requires all training data in one place. Even if we federate the learning, the exchanged models become a new security target, so communication protection must be part of the architecture - not an afterthought.
"@

    # Slide 3: solution.
    $s = $slides[1]
    Add-Header $s "Our response" "Train locally. Learn together. Protect every exchange." "02"
    $cardX = @(44, 266, 488, 710)
    $cardTitles = @("Local ownership", "Shared updates", "Collaborative model", "Protected exchange")
    $cardBodies = @(
        "Each client learns from its own isolated partition.",
        "Model updates move; raw training rows do not.",
        "The coordinator combines all five contributions.",
        "Identity, encryption and replay checks guard messages."
    )
    $cardCodes = @("LOCAL DATA", "MODEL ONLY", "FEDAVG", "PQC + AES")
    $accents = @($C.Teal, $C.Blue, $C.Teal, $C.Amber)
    for ($i = 0; $i -lt 4; $i++) {
        [void](Add-Box $s $cardX[$i] 142 198 250 $C.Panel $C.Border 1 0)
        [void](Add-Circle $s ($cardX[$i]+18) 162 36 $C.Navy2 $accents[$i] 1.4)
        [void](Add-Text $s ("0"+($i+1)) ($cardX[$i]+18) 162 36 36 11 $accents[$i] -Bold -Align 2 -VAlign 3)
        Add-Pill $s $cardCodes[$i] ($cardX[$i]+68) 167 104 22 $C.Navy2 $accents[$i] $accents[$i] 8.5
        [void](Add-Text $s $cardTitles[$i] ($cardX[$i]+18) 222 162 44 17 $C.White -Bold -Align 2 -VAlign 3)
        [void](Add-Text $s $cardBodies[$i] ($cardX[$i]+22) 282 154 68 11 $C.Muted -Align 2)
    }
    [void](Add-Line $s 143 414 809 414 $C.Border 1 0)
    foreach ($x in @(143,365,587,809)) { [void](Add-Circle $s ($x-5) 409 10 $C.Teal $C.Teal 0) }
    [void](Add-Box $s 126 438 708 48 $C.Navy2 $C.Teal 1.4 0)
    [void](Add-Text $s "WORKING PROTOTYPE" 146 450 132 20 10 $C.Teal -Bold)
    [void](Add-Text $s "5 client containers  +  1 coordinator  +  real inference  +  visible security events" 286 448 525 22 12 $C.White -Bold -Align 2)
    Add-Notes $s @"
Timing: 70 seconds.
Federated learning means moving the model to the data rather than moving all data to the model. Each logical client trains locally, and only its learned update participates in Federated Averaging. Our secure mode then protects those application messages: post-quantum algorithms establish keys and authenticate identities, while AES-GCM protects the actual model payloads. The dashboard makes the detection, client participation, rounds and rejection events visible.
"@

    # Slide 4: architecture.
    $s = $slides[2]
    Add-Header $s "Methodology" "One complete learning and security cycle" "03"
    [void](Add-Text $s "5 LOGICAL CLIENTS" 44 112 188 16 9 $C.Teal -Bold)
    [void](Add-Text $s "SECURITY GATE" 304 112 220 16 9 $C.Amber -Bold -Align 2)
    [void](Add-Text $s "FEDAVG COORDINATOR" 684 112 232 16 9 $C.Blue -Bold -Align 2)

    $clientRows = @(
        @("CLIENT 1", "1,230", 0.51),
        @("CLIENT 2", "1,046", 0.75),
        @("CLIENT 3", "591",   0.75),
        @("CLIENT 4", "2,027", 0.90),
        @("CLIENT 5", "1,254", 0.51)
    )
    for ($i=0; $i -lt 5; $i++) {
        $y = 140 + ($i*54)
        [void](Add-Box $s 44 $y 188 43 $C.Panel $C.Border 1 0)
        Add-DroneIcon $s 64 ($y+21) 0.45 $C.Teal
        [void](Add-Text $s $clientRows[$i][0] 88 ($y+7) 72 16 10 $C.White -Bold)
        [void](Add-Text $s ($clientRows[$i][1] + " rows") 88 ($y+23) 74 12 8.3 $C.Muted)
        [void](Add-Box $s 166 ($y+12) 52 7 $C.Track $C.Track 0 0)
        [void](Add-Box $s 166 ($y+12) (52*[double]$clientRows[$i][2]) 7 $C.Red $C.Red 0 0)
        [void](Add-Text $s "attack mix" 166 ($y+24) 52 10 7 $C.Muted2 -Align 2)
    }
    Add-Pill $s "LOCAL DATA STAYS HERE" 58 422 158 22 $C.Navy2 $C.Teal $C.Teal 8.5

    [void](Add-Box $s 294 140 250 304 $C.Panel $C.Amber 1.5 0)
    [void](Add-Text $s "PROTECTED MODEL EXCHANGE" 314 156 210 22 12 $C.White -Bold -Align 2)
    $securityRows = @(
        @("ML-KEM-768", "establish key material"),
        @("HKDF-SHA-256", "derive directional keys"),
        @("ML-DSA-65", "authenticate identities"),
        @("AES-256-GCM", "protect model messages")
    )
    for ($i=0; $i -lt 4; $i++) {
        $y=198+($i*48)
        [void](Add-Circle $s 316 ($y+1) 20 $C.Navy2 $C.Amber 1)
        [void](Add-Text $s ("0"+($i+1)) 316 ($y+1) 20 20 7.5 $C.Amber -Bold -Align 2 -VAlign 3)
        [void](Add-Text $s $securityRows[$i][0] 346 $y 166 15 9.5 $C.White -Bold)
        [void](Add-Text $s $securityRows[$i][1] 346 ($y+16) 166 13 8.3 $C.Muted)
    }
    [void](Add-Line $s 314 395 524 395 $C.Border 1)
    [void](Add-Text $s "Run | round | sender | recipient | type | sequence" 314 405 210 16 7.8 $C.Muted -Align 2)
    [void](Add-Text $s "Tamper and replay checks happen before model decoding" 314 422 210 14 7.8 $C.Amber -Bold -Align 2)

    [void](Add-Box $s 674 140 242 304 $C.Panel $C.Blue 1.5 0)
    Add-ServerIcon $s 772 163 0.9 $C.Blue
    [void](Add-Text $s "CONTROL CENTER" 700 218 190 22 14 $C.White -Bold -Align 2)
    $steps = @("Authenticate", "Reject invalid / replayed", "Decrypt + validate", "Sample-weighted FedAvg", "Distribute next model")
    for ($i=0; $i -lt $steps.Count; $i++) {
        $y=257+($i*31)
        [void](Add-Circle $s 703 ($y+1) 17 $C.Navy2 $C.Blue 1)
        [void](Add-Text $s (""+($i+1)) 703 ($y+1) 17 17 7.5 $C.Blue -Bold -Align 2 -VAlign 3)
        [void](Add-Text $s $steps[$i] 730 $y 157 18 9.2 $C.White -Bold)
    }

    [void](Add-Line $s 232 170 294 170 $C.Teal 2 0 -ArrowEnd)
    [void](Add-Text $s "local updates" 233 148 60 14 7.8 $C.Muted -Align 2)
    [void](Add-Line $s 544 170 674 170 $C.Teal 2 0 -ArrowEnd)
    [void](Add-Line $s 674 410 544 410 $C.Blue 2 0 -ArrowEnd)
    [void](Add-Line $s 294 410 232 410 $C.Blue 2 0 -ArrowEnd)
    [void](Add-Text $s "protected global model" 548 415 122 14 7.8 $C.Muted -Align 2)

    [void](Add-Box $s 44 462 872 38 $C.Navy2 $C.Border 1 0)
    [void](Add-Text $s "MODEL" 60 474 50 12 8 $C.Muted2 -Bold)
    [void](Add-Text $s "15 inputs  ->  128  ->  64  ->  32  ->  Normal / Attack" 112 469 314 20 10 $C.White -Bold)
    [void](Add-Line $s 448 469 448 491 $C.Border 1)
    [void](Add-Text $s "TRAINING" 466 474 58 12 8 $C.Muted2 -Bold)
    [void](Add-Text $s "30 research rounds  |  2 local epochs  |  3-round container demo" 528 469 366 20 9.5 $C.White -Bold)
    [void](Add-Text $s "Scope: source-based logical clients, not verified physical UAVs; the training-only preprocessor was centrally fitted." 238 503 678 10 7.3 $C.Muted2 -Align 3)
    Add-Notes $s @"
Timing: 120 seconds.
Walk from left to right. These are five source-based logical clients with naturally different attack prevalence. Each uses the same 15-feature preprocessing and binary MLP, trains for two local epochs, and produces an update. ML-KEM establishes shared material, HKDF derives directional AES keys, ML-DSA authenticates provisioned identities, and AES-GCM protects each model message. Bound metadata and strict sequences reject wrong-client, wrong-round and replayed messages before model decoding. The coordinator then validates and aggregates updates by sample count. Research used 30 rounds; the container demo deliberately uses three for presentation speed.
Important limitation: the shared preprocessor was fitted centrally on pooled training-client features, so do not claim private federated preprocessing.
"@

    # Slide 5: results.
    $s = $slides[3]
    Add-Header $s "Verified evidence" "Federation helps. Secure transport preserves aggregation." "04"
    [void](Add-Text $s "LOCKED-TEST MACRO-F1" 48 119 430 18 10 $C.Muted2 -Bold)
    [void](Add-Text $s "Balanced performance across Normal and Attack" 48 138 430 15 9 $C.Muted)

    $barData = @(
        @("LOCAL-ONLY MEAN", 89.69, $C.Muted2),
        @("FEDERATED FEDAVG", 95.02, $C.Teal),
        @("CENTRALIZED", 97.75, $C.Blue)
    )
    for ($i=0; $i -lt 3; $i++) {
        $y=184+($i*76)
        [void](Add-Text $s $barData[$i][0] 48 $y 150 16 9.2 $C.White -Bold)
        [void](Add-Text $s (("{0:N2}%" -f $barData[$i][1])) 385 $y 80 18 13 $barData[$i][2] -Bold -Align 3)
        [void](Add-Box $s 48 ($y+24) 416 16 $C.Track $C.Track 0 0)
        [void](Add-Box $s 48 ($y+24) (416*[double]$barData[$i][1]/100) 16 $barData[$i][2] $barData[$i][2] 0 0)
    }
    Add-Pill $s "+5.33 points vs average isolated model" 48 414 250 25 $C.Panel2 $C.Teal $C.Teal 9
    Add-Pill $s "-2.73 points vs centralized ceiling" 306 414 158 25 $C.Panel2 $C.Blue $C.Blue 8.4

    $metricCards = @(
        @(510,130,190,116,"97.67%","ATTACK RECALL","97.80% precision",$C.Teal),
        @(716,130,200,116,"7.49%","FALSE-POSITIVE RATE","Reported - not hidden",$C.Amber),
        @(510,262,190,116,"0.0","AGGREGATION DIFFERENCE","secure vs plain | tol. 1e-7",$C.Blue),
        @(716,262,200,116,"12","INVALID MESSAGES REJECTED","0 changed aggregation",$C.Red)
    )
    foreach ($m in $metricCards) {
        [void](Add-Box $s $m[0] $m[1] $m[2] $m[3] $C.Panel $C.Border 1 0)
        [void](Add-Text $s $m[4] ($m[0]+16) ($m[1]+14) ($m[2]-32) 36 23 $m[7] -Bold)
        [void](Add-Text $s $m[5] ($m[0]+16) ($m[1]+55) ($m[2]-32) 17 8.8 $C.White -Bold)
        [void](Add-Text $s $m[6] ($m[0]+16) ($m[1]+78) ($m[2]-32) 24 8.5 $C.Muted)
    }
    [void](Add-Box $s 510 394 406 45 $C.Navy2 $C.Teal 1 0)
    [void](Add-Text $s "53,949 locked-test flows" 526 405 174 18 10 $C.White -Bold)
    [void](Add-Text $s "|  all 5 clients completed 3 secure demo rounds" 700 405 198 18 9.3 $C.Muted)
    [void](Add-Text $s "Federation helped versus isolated learning, while centralized pooling remained the accuracy upper baseline." 48 463 868 25 12 $C.White -Bold -Align 2)
    [void](Add-Text $s "Source: locked UAVIDS model evaluation and verified secure-demo artifacts. Host-specific timing is not used as a general performance claim." 48 494 868 12 7.3 $C.Muted2 -Align 2)
    Add-Notes $s @"
Timing: 100 seconds.
Macro-F1 gives equal importance to the Normal and Attack classes, so it is more informative here than accuracy alone. FedAvg reached 95.02 percent on the locked unseen-source test - 5.33 percentage points above the average local-only model, although centralized pooling remained 2.73 points higher. The security layer did not alter learning: secure and plain aggregation matched exactly within the verifier's tolerance. Twelve controlled malformed, tampered, replayed, wrong-identity or wrong-context messages were safely rejected, and none entered aggregation. The 7.49 percent false-positive rate is a real limitation and should be stated openly.
"@

    # Slide 6: demo transition.
    $s = $slides[4]
    Add-Header $s "Live demonstration" "From traffic pattern to security event" "05"
    [void](Add-Text $s "Watch the same frozen model respond as the traffic profile changes." 44 112 640 20 12 $C.Muted)

    $demoStages = @(
        @(44,154,246,"01","NOMINAL LINK","Normal verdict","Low, stable traffic features",$C.Green),
        @(357,154,246,"02","TRANSITIONAL","Decision boundary","A deliberately mixed profile",$C.Amber),
        @(670,154,246,"03","SUSTAINED FLOOD","Attack verdict","Alert and counters update",$C.Red)
    )
    foreach ($d in $demoStages) {
        [void](Add-Box $s $d[0] $d[1] $d[2] 148 $C.Panel $C.Border 1 0)
        [void](Add-Circle $s ($d[0]+18) ($d[1]+18) 28 $C.Navy2 $d[7] 1.3)
        [void](Add-Text $s $d[3] ($d[0]+18) ($d[1]+18) 28 28 9 $d[7] -Bold -Align 2 -VAlign 3)
        [void](Add-Text $s $d[4] ($d[0]+58) ($d[1]+23) ($d[2]-76) 18 9 $C.White -Bold)
        [void](Add-Text $s $d[5] ($d[0]+18) ($d[1]+67) ($d[2]-36) 26 17 $d[7] -Bold)
        [void](Add-Text $s $d[6] ($d[0]+18) ($d[1]+105) ($d[2]-36) 24 9.5 $C.Muted)
    }
    [void](Add-Line $s 296 228 348 228 $C.Teal 2 0 -ArrowEnd)
    [void](Add-Line $s 609 228 661 228 $C.Teal 2 0 -ArrowEnd)

    [void](Add-Text $s "WHAT THE DASHBOARD PROVES" 44 336 250 16 9.5 $C.Teal -Bold)
    $badges = @(
        @(44,365,196,"REAL MODEL","Frozen checkpoint + hash"),
        @(254,365,196,"BINARY VERDICT","Normal / Attack + confidence"),
        @(464,365,196,"5 CLIENTS","Rounds + update progress"),
        @(674,365,242,"SECURE MODE","Authentication + rejections")
    )
    foreach ($b in $badges) {
        [void](Add-Box $s $b[0] $b[1] $b[2] 66 $C.Navy2 $C.Border 1 0)
        [void](Add-Text $s $b[3] ($b[0]+13) ($b[1]+12) ($b[2]-26) 17 9 $C.White -Bold)
        [void](Add-Text $s $b[4] ($b[0]+13) ($b[1]+34) ($b[2]-26) 18 8.7 $C.Muted)
    }
    [void](Add-Box $s 44 452 872 42 $C.Teal $C.Teal 0 0)
    [void](Add-Text $s "SWITCHING TO THE LIVE OPERATIONS DASHBOARD  ->" 44 452 872 42 14 $C.Navy -Bold -Align 2 -VAlign 3)
    [void](Add-Text $s "Honest boundary: generated model inputs - not packet capture; binary detection only; live/replay status remains visible." 44 501 872 12 7.5 $C.Muted2 -Align 2)
    Add-Notes $s @"
Timing: approximately 180 seconds, leaving around 50 seconds of buffer.
Demo sequence: first verify that the backend reports the real frozen model identifier. Start the nominal profile and point out the Normal verdict and counters. Move to Transitional to show that the detector has a real decision boundary rather than a rigged switch. Then select Sustained Flood and show the Attack verdict, confidence, alert and counters. Finally, scroll to the federated and security panels: five clients, round progress, secure mode, authenticated identities and any rejection events.
Be explicit that the injector produces internally consistent model-feature vectors; it is not live packet capture. The model is binary and does not name an attack family. If Docker telemetry fails, use verified recorded telemetry while keeping the replay indicator visible; real frozen-model inference should still remain active.
"@

    # Remove unused layouts/master artifacts only by saving the edited copy.
    $presentation.Save()
} finally {
    if ($presentation -ne $null) { $presentation.Close() }
    $ppt.Quit()
}

Write-Output $outputFull
