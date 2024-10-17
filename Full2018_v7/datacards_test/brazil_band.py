import ROOT
from ROOT import TGraph, TCanvas, TLegend, gPad, TLine

ADD_OBSERVED = False  # Set this to True to add the observed limit

def getLimits(filename):
    limits = []
    with open(filename) as f:
        for line in f:
            if "Observed Limit" in line:
                limits.append(float(line.split()[-1]))
            elif "Expected  2.5%" in line:
                limits.append(float(line.split()[-1]))
            elif "Expected 16.0%" in line:
                limits.append(float(line.split()[-1]))
            elif "Expected 50.0%" in line:
                limits.append(float(line.split()[-1]))
            elif "Expected 84.0%" in line:
                limits.append(float(line.split()[-1]))
            elif "Expected 97.5%" in line:
                limits.append(float(line.split()[-1]))
    return limits

def plotUpperLimits(labels, values, mhs, mDM):
    ROOT.gROOT.SetBatch(True)
    
    N = len(labels)
    yellow = TGraph(2 * N)    # yellow band
    green = TGraph(2 * N)     # green band
    median = TGraph(N)        # median line
    observed = TGraph(N)      # observed line if ADD_OBSERVED is True

    up2s = []
    if ADD_OBSERVED:
        for i in range(N):
            file_name = labels[i]
            limit = getLimits(file_name)
            up2s.append(limit[5])
            yellow.SetPoint(i, values[i], limit[5]) # + 2 sigma
            green.SetPoint(i, values[i], limit[4])  # + 1 sigma
            median.SetPoint(i, values[i], limit[3]) # median
            green.SetPoint(2 * N - 1 - i, values[i], limit[2]) # - 1 sigma
            yellow.SetPoint(2 * N - 1 - i, values[i], limit[1]) # - 2 sigma
            observed.SetPoint(i, values[i], limit[0])
    else:
        for i in range(N):
            file_name = labels[i]
            limit = getLimits(file_name)
            up2s.append(limit[4])
            yellow.SetPoint(i, values[i], limit[4]) # + 2 sigma
            green.SetPoint(i, values[i], limit[3])  # + 1 sigma
            median.SetPoint(i, values[i], limit[2]) # median
            green.SetPoint(2 * N - 1 - i, values[i], limit[1]) # - 1 sigma
            yellow.SetPoint(2 * N - 1 - i, values[i], limit[0]) # - 2 sigma
    
    W = 800
    H = 600
    T = 0.08 * H
    B = 0.12 * H
    L = 0.12 * W
    R = 0.04 * W
    c = TCanvas("c", "c", 100, 100, W, H)
    c.SetFillColor(0)
    c.SetBorderMode(0)
    c.SetFrameFillStyle(0)
    c.SetFrameBorderMode(0)
    c.SetLeftMargin(L / W)
    c.SetRightMargin(R / W)
    c.SetTopMargin(T / H)
    c.SetBottomMargin(B / H)
    c.SetTickx(0)
    c.SetTicky(0)
    c.SetGrid()
    c.SetLogy()
    c.cd()
    frame = c.DrawFrame(min(values)*0.9, 0, max(values)*1.1, max(up2s)*1.1)
    frame.GetYaxis().CenterTitle()
    frame.GetYaxis().SetTitleSize(0.05)
    frame.GetXaxis().SetTitleSize(0.05)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetTitleOffset(0.9)
    frame.GetXaxis().SetNdivisions(508)
    frame.GetYaxis().CenterTitle(True)
    frame.GetYaxis().SetTitle("95% CL limit on #sigma / #sigma_{SM}")
    frame.GetXaxis().SetTitle("m_{Z'} [GeV]")
    frame.SetMinimum(0.1)
    #frame.SetMaximum(max(up2s) * 1.05)
    frame.SetMaximum(10)
    frame.GetXaxis().SetLimits(min(values), max(values))

    yellow.SetFillColor(ROOT.kOrange)
    yellow.SetLineColor(ROOT.kOrange)
    yellow.SetFillStyle(1001)
    yellow.Draw('F')

    green.SetFillColor(ROOT.kGreen + 1)
    green.SetLineColor(ROOT.kGreen + 1)
    green.SetFillStyle(1001)
    green.Draw('Fsame')

    median.SetLineColor(1)
    median.SetLineWidth(2)
    median.SetLineStyle(2)
    median.Draw('Lsame')

    if ADD_OBSERVED:
        observed.SetLineColor(1)  # black color
        observed.SetLineWidth(2)
        observed.SetLineStyle(1)  # solid line
        observed.SetMarkerStyle(20)  # full circle
        observed.Draw('LPsame')  # L for line, P for points

    # Draw red line at y = 1
    line = TLine(min(values), 1, max(values), 1)
    line.SetLineColor(ROOT.kRed)
    line.SetLineWidth(1)
    line.Draw('same')
    
    # Add the CMS Preliminary label in the upper left corner
    lumi = ROOT.TLatex()
    lumi.SetNDC()
    lumi.SetTextFont(42)
    lumi.SetTextSize(0.07)
    lumi.SetTextAlign(11)
    lumi.DrawLatex(0.12, 0.935, "Private")

    # Add the luminosity label in the upper right corner
    lumi.SetTextSize(0.04)
    lumi.SetTextAlign(31)
    lumi.DrawLatex(0.92, 0.935, "59.7 fb^{-1} (13 TeV)")

    ROOT.gPad.SetTicks(1, 1)
    frame.Draw('sameaxis')

    legend = TLegend(0.51, 0.15, 0.89, 0.35)  # Bottom right 
    legend.SetFillColor(ROOT.kWhite)
    legend.SetBorderSize(1)  # 1 for visible border
    legend.SetLineColor(ROOT.kBlack)  # Border color
    legend.SetTextSize(0.041)
    legend.SetTextFont(32)
    legend.AddEntry(median, "Expected", 'L')
    legend.AddEntry(green, "#pm 1 std. deviation", 'f')
    legend.AddEntry(yellow, "#pm 2 std. deviation", 'f')
    if ADD_OBSERVED:
        legend.AddEntry(observed, "Observed Limit", 'LP')
    legend.SetHeader(f'm_{{s}} = {mhs} GeV, m_{{#chi}} = {mDM} GeV')
    legend.Draw()

    c.SaveAs(f"BrazilBplot_mx{mDM}_ms{mhs}.png")
    c.Close()
