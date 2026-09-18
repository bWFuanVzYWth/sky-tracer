"""Compare actual demo previews at identical views and display settings."""
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path("out")
fig,axes=plt.subplots(2,4,figsize=(16,6.6))
for row,case in enumerate(["noon_horizon_85","noon_wide_85"]):
    old=Image.open(root/"noon_validation_v3"/f"lut_{case}.png")
    new=Image.open(root/"noon_validation_v4_rgb"/f"lut_{case}.png")
    w,h=new.width//4,new.height
    for col,(source,panel,label) in enumerate([(old,0,"v3 / 65 solar nodes"),(new,0,"v4 / 97 nodes / Rec.2020"),(new,1,"PT / 8192 spp"),(new,2,"v4 absolute difference x4")]):
        axes[row,col].imshow(source.crop((panel*w,0,(panel+1)*w,h)))
        axes[row,col].set_axis_off()
        if row==0:axes[row,col].set_title(label,fontsize=11)
fig.suptitle("Sun elevation 85 degrees, observer 200 m\nActual demo display: horizon view (top), wide view (bottom)",fontsize=14)
fig.tight_layout(pad=.7)
fig.savefig(root/"noon_demo_comparison.png",dpi=150,bbox_inches="tight")
