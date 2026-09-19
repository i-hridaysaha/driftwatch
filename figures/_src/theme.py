"""Visual theme, z-order contract, and drawing primitives for case-study figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---- palette (mirrors the site design tokens; see Wix/Homepage/theme) ----
INK       = "#0F0E0E"   # --ink, near-black text
BODY      = "#2B2A2A"   # --body, long-form text
BAND      = "#0E0D0D"   # --band, the darkest surface: hero and banner background
BAND2     = "#1A1917"   # one step up from the band, for tiles sitting on it
VOLT      = "#FF5701"   # --volt, the signature accent
VOLT_TXT  = "#FF6A2A"   # --volt-hover (dark), accent TEXT on the band
VOLT_DEEP = "#8A3200"   # accent text on a light tint, where VOLT itself fails contrast
VOLT_SOFT = "#FFE7D8"   # emphasis fill on light, one step down from --brand-tint
VOLT_EDGE = "#FFB88F"   # border of the single emphasised block
VOLT_PRESS= "#CC4400"   # --volt-press, edge on a solid VOLT fill
PAPER     = "#FFFFFF"   # --card
CARD      = "#FBFBF7"   # --paper, the figure canvas
SURFACE   = "#F0EFE9"   # --surface, inert fills and empty bar tracks
BORDER    = "#D2D0C8"   # --line
BORDER_D  = "#B9B7AE"   # one step down from --line, for node edges
MUTE      = "#5A5A5A"   # --secondary, captions and sublabels
FAINT     = "#8F8B83"
RED       = "#B3261E"   # --red: bad, leak, reject
RED_SOFT  = "#F9E4E3"   # pushed pink, so it never reads as a wash of the warm accent
GREEN     = "#1E8E4A"   # --green: good, fix, accept
GREEN_SOFT= "#E4F1E8"
# grouping tints. Warm is reserved for the accent, so the third plane is sand, not cream.
BLUE_T  = "#EAF1F8"; BLUE_E  = "#C6D8EC"
SAND_T  = "#F2F1EA"; SAND_E  = "#DCD9CE"
PINK_T  = "#FAE7E9"; PINK_E  = "#E6C3C8"
GREEN_T = "#EDF6EC"; GREEN_E = "#CBE3CC"
# categorical series colours. None of these is VOLT: the accent marks the one thing
# a figure is about, so a mere category may never wear it.
C_GRAY="#B7BABE"; C_BLUE="#4F86C6"; C_GREEN="#3E8F58"; C_VIOLET="#7C5CBF"; C_RED="#C0524C"

# ---- z-order contract ----
# Arrows carry the flow, so they render above every node and band.
# Never override these ad hoc: an arrow hidden behind a box is an unreadable claim.
Z_BAND      = 1   # grouping bands, background panels
Z_PANEL     = 2   # secondary panels and fills
Z_NODE      = 3   # node boxes
Z_NODE_TEXT = 5   # text inside nodes
Z_ARROW     = 6   # connector lines
Z_HEAD      = 7   # arrowheads
Z_LABEL     = 8   # connector labels, always on top

# ---- hero safe zone (measured from the live Wix template) ----
# Canvas y runs 0 at the BOTTOM to ax._Y at the TOP, so the desktop title block,
# which covers the bottom of the image, sets a MINIMUM y for safe content.
HERO_PX         = (2400, 1000)   # author at this size
HERO_ASPECT     = 2.4
HERO_SAFE_X     = (27.0, 73.0)   # canvas units: centre 46 percent survives the mobile crop
HERO_SAFE_Y_MIN = 0.28           # content must sit ABOVE this fraction of the height

def use_theme():
    for fam in ["Helvetica", "Arial", "DejaVu Sans"]:
        if any(f.name == fam for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams.update({
        "font.size": 11, "text.color": INK,
        "axes.edgecolor": BORDER_D, "axes.labelcolor": INK,
        "xtick.color": MUTE, "ytick.color": MUTE,
        "axes.linewidth": 0.9, "figure.dpi": 200,
        "savefig.dpi": 200,
    })

MONO = "DejaVu Sans Mono"

# ---------- primitives ----------
def canvas(w=7.2, h=4.6, bg=CARD):
    """Blank figure with a 0..100 x-axis. y-axis is scaled to aspect, top is ax._Y."""
    fig = plt.figure(figsize=(w, h)); fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w); ax.axis("off"); ax.set_facecolor(bg)
    ax._Y = 100 * h / w
    return fig, ax

def rbox(ax, x, y, w, h, fc=PAPER, ec=BORDER_D, lw=1.1, r=2.2, z=Z_NODE, alpha=1):
    # A rounding radius larger than half the shorter side turns the box into a lens.
    # Bar-style calls pass a data-driven width, so clamp rather than trust the caller.
    r = min(r, abs(w)/2, abs(h)/2)
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha, mutation_aspect=1)
    ax.add_patch(p); return p

def pill(ax, x, y, w, h, fc=INK, ec="none", r=None, z=Z_NODE):
    r = r if r else h/2
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.2, zorder=z))

def track(ax, x, y, s, size=8.5, color=MUTE, weight="bold", ha="left"):
    """Letter-tracked uppercase kicker."""
    ax.text(x, y, " ".join(list(s.upper())).replace("     ", "   "), fontsize=size,
            color=color, fontweight=weight, ha=ha, va="center", zorder=Z_LABEL)

def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.6, style="-|>", ms=15, ls="-", z=Z_ARROW):
    """Connector. Defaults to Z_ARROW so it always renders above nodes and bands."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=ms, color=color, lw=lw, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0))

def oarrow(ax, pts, color=INK, lw=2.0, ms=16, ls="-", z=Z_ARROW):
    """Orthogonal multi-segment connector. Arrowhead sits on the final segment.

    Route long connectors through an empty margin using intermediate points rather
    than drawing a diagonal across content.
    """
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z,
                solid_capstyle="round", solid_joinstyle="round")
    arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1],
          color=color, lw=lw, ms=ms, z=Z_HEAD, ls=ls)

def alabel(ax, x, y, s, color=INK, size=8.2, ha="left", weight="normal", rot=0):
    """Connector label. Always on top so it is never clipped by a node."""
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
            zorder=Z_LABEL, style="italic", fontweight=weight, rotation=rot)

def chip(ax, x, y, s, fc=VOLT, tc=INK, size=9.5, pad=1.4, weight="bold", h=None, ec="none"):
    w = pad*2 + len(s)*size*0.105
    h = h or (size*0.32)
    rbox(ax, x, y-h/2, w, h, fc=fc, ec=ec, lw=1, r=h/2, z=Z_NODE+1)
    ax.text(x+w/2, y, s, fontsize=size, color=tc, fontweight=weight,
            ha="center", va="center", zorder=Z_NODE_TEXT)
    return w

def title_block(ax, x, y, title, sub=None, tsize=15):
    ax.text(x, y, title.upper(), fontsize=tsize, color=INK, fontweight="bold",
            ha="left", va="center", zorder=Z_LABEL)
    if sub:
        ax.text(x, y-tsize*0.28, sub, fontsize=9.5, color=MUTE, style="italic",
                ha="left", va="center", zorder=Z_LABEL)

def safe_box(ax, show=True):
    """Draw the Wix hero safe box. Render once with this on, confirm, then turn it off.

    The box spans the centre 46 percent of width (mobile crop) and the top 72 percent
    of height (desktop title overlay). Anything outside is invisible to some readers.
    """
    if not show:
        return
    Y = ax._Y
    x0, x1 = HERO_SAFE_X
    y0 = Y * HERO_SAFE_Y_MIN
    ax.add_patch(Rectangle((x0, y0), x1-x0, Y-y0, fill=False, ec="#FF2D9B",
                           lw=1.4, ls=(0, (5, 4)), zorder=Z_LABEL))
    ax.text(x0, y0-1.8, "SAFE BOX: everything outside is cropped or covered",
            fontsize=7.5, color="#FF2D9B", zorder=Z_LABEL)

def save(fig, path):
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print("wrote", path)