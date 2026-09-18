"""Compare the actual four-wave GPU integrator with the frozen spectral teacher."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from preview_wavelengths_cpu import display, heatmap, save_png

F = np.float32

def stats(values):
    return dict(zip(('p50','p95','p99','max'), map(float,np.percentile(values,[50,95,99,100]))))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('out/wavelength_search_dataset_v1'))
    p.add_argument('--runs',type=Path,default=Path('out/four_wave_validation_v1'))
    p.add_argument('--candidate',type=Path,default=Path('out/wavelength_counts_v1/N4_best.json'))
    a=p.parse_args()
    meta=json.loads((a.dataset/'dataset.json').read_text())
    plan=json.loads((a.dataset/'queries.json').read_text())
    c=json.loads(a.candidate.read_text())
    total=np.fromfile(a.dataset/'total.f32',dtype=F).reshape(meta['array_shape']).T
    W=np.array(meta['rgb_from_integrated'],dtype=F)
    W4=np.array(c['rgb_from_integrated'],dtype=F)
    folder=a.runs/'visual';folder.mkdir(exist_ok=True)
    rows=[];html=[]
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    for scene in plan['images']:
        name=scene['name'];w=scene['width'];h=scene['height'];ids=slice(scene['start'],scene['end'])
        teacher=total[ids]@W
        quadrature=total[ids][:,c['indices']]@W4
        norm=np.linalg.norm(teacher,axis=1);active=norm>F(1e-8);den=np.maximum(norm,F(1e-7))
        images={str(k):np.fromfile(a.runs/f'{name}_{k}.f32',dtype=F).reshape(-1,4)[:,:3] for k in (32,64,128,256,512)}
        assert all(np.isfinite(x).all() for x in images.values())
        error=lambda x,y:stats((F(100)*np.linalg.norm(x-y,axis=1)/den)[active])
        row=dict(scene=name,active=int(active.sum()),total=len(norm),against_41={k:error(v,teacher) for k,v in images.items()},against_4={k:error(v,quadrature) for k,v in images.items()},step_convergence={k:error(v,images['512']) for k,v in images.items() if k!='512'})
        rows.append(row)
        lum=teacher@np.array([.2627,.6780,.0593],dtype=F)
        exposure=F(.6)/max(np.quantile(lum[lum>F(1e-8)],F(.9)),F(1e-7))
        sheet=Image.new('RGB',(4*384,240),(20,24,30));draw=ImageDraw.Draw(sheet)
        for i,(key,title,values) in enumerate([
            ('teacher','Frozen 41-band LUT',display(teacher,exposure)),
            ('four','Same LUT, 4 wavelengths',display(quadrature,exposure)),
            ('runtime','4-wave + anisotropic MS / 128',display(images['128'],exposure)),
            ('error','Error vs 41-band: black 0, yellow 10%',heatmap(np.linalg.norm(images['128']-teacher,axis=1)/den,10)),
        ]):
            save_png(folder/f'{name}_{key}.png',values,w,h)
            sheet.paste(Image.open(folder/f'{name}_{key}.png').resize((384,192)),(i*384,48))
            draw.text((i*384+4,6),title,font=font,fill='white')
        draw.text((2*384+4,27),f"P95 {row['against_41']['128']['p95']:.3f}%",font=font,fill='#cbd5e1')
        sheet.save(folder/f'{name}_sheet.png')
        html.append(f'<section><h2>{name}</h2><p>h={scene["altitude_km"]} km, Sun {scene["sun_elevation_deg"]:.3f}°. P95={row["against_41"]["128"]["p95"]:.3f}%</p><img src="{name}_sheet.png"></section>')
    (a.runs/'summary.json').write_text(json.dumps(dict(rows=rows,metric='Linear Rec.2020 vector relative error, norm floor 1e-7; active norm >1e-8',limitations=['Full rendering error, not isolated spectral error','Reference interpolation/transport bias remains, especially high altitude','Static views do not establish temporal stability']),indent=2))
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Four-wave anisotropic prototype</title><style>body{background:#10151d;color:#ddd;font:16px system-ui;margin:24px}img{width:100%;max-width:1536px}section{margin:28px 0}p{max-width:1200px;line-height:1.6}</style><h1>四波长 + 各向异性多重散射</h1><p>实际 GPU 积分，128 步；资源 12.69 MiB。依次为冻结 41 波段参考、四波长查询同一参考、新积分器、误差热图。每个场景使用同一曝光，太阳盘不参与数值比较。高空大差异包含参考 LUT 自身的插值/输运残差，不能全部解释为压缩误差。这是精度原型，尚未加入生产 SkyView / froxel 缓存。</p>'+''.join(html),encoding='utf-8')
    for row in rows: print(row['scene'],row['against_41']['128']['p95'],row['step_convergence']['128']['p95'])

if __name__=='__main__':main()
