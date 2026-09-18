"""Independent CPU visual audit after the numerical shortlist is frozen.

Keeps the full 41-band indirect RGB contribution identical for all candidates.
This isolates spectral quadrature; it is NOT a render of the compressed MS LUT.
"""
import argparse
import json
import html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from search_wavelengths_cpu import read_dataset, stats

F = np.float32
TO_SRGB = np.array([[1.6604910,-.5876411,-.0728499],[-.1245505,1.1328999,-.0083494],[-.0181508,-.1005789,1.1187297]], dtype=F)
TO_LMS = np.array([[.4122214708,.5363325363,.0514459929],[.2119034982,.6806995451,.1073969566],[.0883024619,.2817188376,.6299787005]], dtype=F)
LMS_TO_LAB = np.array([[.2104542553,.7936177850,-.0040720468],[1.9779984951,-2.4285922050,.4505937099],[.0259040371,.7827717662,-.8086757660]], dtype=F)
LMS_TO_RGB = np.array([[4.0767416621,-3.3077115913,.2309699292],[-1.2684380046,2.6097574011,-.3413193965],[-.0041960863,-.7034186147,1.7076147010]], dtype=F)
HUE_TO_ROOT = np.array([[.3963377774,.2158037573],[-.1055613458,-.0638541728],[-.0894841775,-1.2914855480]], dtype=F)


def softmin(value, limit, power):
    lower = np.maximum(np.minimum(value, limit), F(0))
    higher = np.maximum(np.maximum(value, limit), F(1e-30))
    return lower * np.power(F(1) + np.power(lower / higher, power), -F(1) / power)


def display(rgb, exposure):
    """f32 port of demo reinhard_gamut.wgsl (overexposure=1.1), then sRGB OETF."""
    srgb = np.maximum(np.maximum(rgb * exposure, F(0)) @ TO_SRGB.T, F(0))
    lab = np.cbrt(srgb @ TO_LMS.T) @ LMS_TO_LAB.T
    light = lab[:, 0]
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    hue = lab[:, 1:] / np.maximum(chroma[:, None], F(1e-30))
    direction = hue @ HUE_TO_ROOT.T
    x = ((F(1.1) / (F(1.1) - F(.18))) ** F(2) - F(1)) / F(.18) * light**3
    invroot = F(1) / np.sqrt(F(1) + x)
    out_light = np.cbrt(F(1.1) * x * invroot * invroot / (F(1) + invroot))
    red = -F(1.88170328) * hue[:, 0] - F(.80936493) * hue[:, 1] > F(1)
    green = (~red) & (F(1.81444104) * hue[:, 0] - F(1.19445276) * hue[:, 1] > F(1))
    kind = np.where(red, 0, np.where(green, 1, 2))
    K = np.array([[1.19086277,1.76576728,.59662641,.75515197,.56771245],
                  [.73956515,-.45954404,.08285427,.12541070,.14503204],
                  [1.35733652,-.00915799,-1.15130210,-.50559606,.00692167]], dtype=F)[kind]
    h0, h1 = hue.T
    saturation = K[:, 0] + K[:, 1]*h0 + K[:, 2]*h1 + K[:, 3]*h0*h0 + K[:, 4]*h0*h1
    row = LMS_TO_RGB[kind]
    root = F(1) + saturation[:, None]*direction
    f = np.sum(row * root**3, axis=1)
    f1 = np.sum(row * F(3)*direction*root**2, axis=1)
    f2 = np.sum(row * F(6)*direction**2*root, axis=1)
    denominator = f1*f1 - F(.5)*f*f2
    saturation -= f*f1 / np.where(np.abs(denominator)>F(1e-30),denominator,F(1))
    alignment = np.maximum(hue @ np.array([-.10362546,-.99461639], dtype=F), F(0))
    saturation = np.minimum(saturation, F(.57) + F(1) - alignment**256)
    root = F(1) + saturation[:, None]*direction
    cusp = np.cbrt(F(1) / np.maximum(np.max(root**3 @ LMS_TO_RGB.T, axis=1), F(1e-30)))
    black_chroma = out_light*saturation
    white_chroma = cusp*saturation*(F(1)-out_light) / np.maximum(F(1)-cusp, F(1e-30))
    root = out_light[:, None] + white_chroma[:, None]*direction
    f = root**3 @ LMS_TO_RGB.T - F(1)
    f1 = (F(3)*direction*root**2) @ LMS_TO_RGB.T
    f2 = (F(6)*direction**2*root) @ LMS_TO_RGB.T
    with np.errstate(invalid='ignore', divide='ignore'):
        reciprocal = f1/(f1*f1-F(.5)*f*f2)
        step = np.where(reciprocal >= F(0), -f*reciprocal, F(1e20))
    white_chroma += np.min(step, axis=1)
    t = np.clip((out_light-cusp)/np.maximum(F(1)-cusp, F(1e-30)), F(0), F(1))
    shoulder = t*(F(1)-t)
    white_chroma *= F(1) - F(.0035)*F(16)*shoulder**2
    cap = np.maximum(softmin(black_chroma, white_chroma, F(4))/np.maximum(out_light, F(1e-30)), F(0))
    cap = np.where(out_light >= F(1), F(0), np.where(out_light <= F(0), saturation, cap))
    desired = chroma/np.maximum(light,F(1e-30))*(F(1)-out_light**12)
    power = F(32)-F(256)*(out_light*(F(1)-out_light))**2
    out_sat = softmin(desired, cap, power)
    out_sat = np.where(chroma <= F(1e-8), F(0), out_sat)
    root = out_light[:, None]*(F(1)+out_sat[:, None]*direction)
    mapped = np.clip(F(.99999)*(root**3 @ LMS_TO_RGB.T),F(0),F(1))
    mapped[light <= F(0)] = F(0)
    assert np.isfinite(mapped).all()
    return np.where(mapped <= F(.0031308), F(12.92)*mapped, F(1.055)*mapped**F(1/2.4)-F(.055))


def heatmap(relative, max_percent=1):
    # Common range across candidates; purple/orange/yellow.
    t = np.clip(relative/(F(max_percent)*F(.01)),F(0),F(1))
    return np.stack([np.clip(F(2)*t,F(0),F(1)), np.clip(F(2)*t-F(1),F(0),F(1)), F(.3)*np.sin(F(np.pi)*t)],axis=-1)


def save_png(file, values, w, h):
    pixels = np.round(np.clip(values,F(0),F(1))*F(255)).astype(np.uint8).reshape(h,w,3)
    Image.fromarray(pixels).save(file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('search',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--heat-max-percent',type=float,default=1)
    a=p.parse_args()
    if not np.isfinite(a.heat_max_percent) or a.heat_max_percent<=0:
        raise ValueError('heat range must be positive and finite')
    if a.out.exists():
        raise ValueError('use a new output directory')
    a.out.mkdir(parents=True)
    search = json.loads((a.search/'search.json').read_text())
    meta, plan, data = read_dataset(Path(search['dataset']))
    assert meta['source_band_checksums'] == search['source_band_checksums']
    W=np.array(meta['rgb_from_integrated'],dtype=F)
    total = data['total']@W
    direct = data['direct']@W
    candidates = search['baselines']+search['selected']
    try:
        font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',15)
        title_font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',19)
    except OSError:
        font=title_font=ImageFont.load_default()
    output=[]
    for spec in plan['images']:
        name=spec['name'];start,end=spec['start'],spec['end'];w,h=spec['width'],spec['height']
        teacher=total[start:end];norm=np.linalg.norm(teacher,axis=1);mask=norm>F(1e-8)
        lum=np.maximum(teacher@np.array([.2627,.6780,.0593],dtype=F),F(0))
        exposure=F(.6)/max(np.quantile(lum[lum>F(1e-8)],F(.9)),F(1e-7))
        shown=display(teacher,exposure)
        save_png(a.out/f'{name}_reference.png',shown,w,h)
        beauty=[Image.open(a.out/f'{name}_reference.png')];heats=[Image.new('RGB',(w,h))]
        titles=['41-band reference'];summaries=['Same indirect RGB for every candidate']
        view=dict(**spec,exposure=float(exposure),candidates=[])
        raw={'reference':teacher.reshape(h,w,3)}
        for c in candidates:
            cid=c['id'];indices=c['indices'];matrix=np.array(c['rgb_from_integrated'],dtype=F)
            error = data['direct'][start:end,indices]@matrix-direct[start:end]
            hybrid = teacher+error
            rgb=display(hybrid,exposure)
            relative=np.linalg.norm(error,axis=1)/np.maximum(norm,F(1e-7))
            relative[~mask]=0
            save_png(a.out/f'{name}_{cid}.png',rgb,w,h)
            save_png(a.out/f'{name}_{cid}_display_diff20.png',np.abs(rgb-shown)*F(20),w,h)
            save_png(a.out/f'{name}_{cid}_signed_diff20.png',F(.5)+(rgb-shown)*F(20),w,h)
            save_png(a.out/f'{name}_{cid}_heat.png',heatmap(relative,a.heat_max_percent),w,h)
            info=dict(id=cid,relative_error_percent=stats(relative[mask]),
                display_max_difference_8bit=float(np.max(np.abs(rgb-shown))*F(255)),
                negative_hybrid_channels=int(np.count_nonzero(hybrid < -F(1e-6))))
            view['candidates'].append(info)
            raw[cid]=hybrid.reshape(h,w,3)
            beauty.append(Image.open(a.out/f'{name}_{cid}.png'))
            heats.append(Image.open(a.out/f'{name}_{cid}_heat.png'))
            titles.append(cid+' / '+','.join(str(int(x)) for x in c['wavelengths_nm']))
            summaries.append(f"P95 {info['relative_error_percent']['p95']:.3f}%  max {info['relative_error_percent']['max']:.3f}%")
        np.savez_compressed(a.out/f'{name}_linear_rec2020.npz',**raw)
        output.append(view)
        cellw,cellh=384,436
        rows=(len(beauty)+2)//3
        sheet=Image.new('RGB',(cellw*3,rows*cellh+64),(21,25,31));draw=ImageDraw.Draw(sheet)
        draw.text((8,6),f'{name}: {spec["altitude_km"]} km / Sun {spec["sun_elevation_deg"]:.2f} deg',font=title_font,fill='white')
        draw.text((8,32),f'Top: same exposure. Bottom: RGB error; black 0%, orange {a.heat_max_percent/2:g}%, yellow >={a.heat_max_percent:g}%.',font=font,fill='#bbc3cc')
        for i,(im,hm) in enumerate(zip(beauty,heats)):
            x=(i%3)*cellw;y=64+(i//3)*cellh
            draw.text((x+4,y+2),titles[i],font=font,fill='white')
            sheet.paste(im.resize((384,192)),(x,y+24))
            draw.text((x+4,y+218),summaries[i],font=font,fill='#bbc3cc')
            sheet.paste(hm.resize((384,192)),(x,y+240))
        sheet.save(a.out/f'{name}_sheet.png')
    result=dict(kind='independent_wavelength_visual_audit_v1',search=str(a.search),scenes=output,heat_max_percent=a.heat_max_percent,
        note='Shortlist frozen before visual scoring. Hybrid=Ltotal41+(S1+B)N-(S1+B)41; unchanged ideal indirect RGB, no compressed MS or runtime march error. No direct solar disk. Per-view reference-based exposure, fixed across candidates. Demo Enhanced Reinhard 1.1 f32 port, sRGB preview. Dense views are an audit, not a statistical guarantee.')
    (a.out/'visual.json').write_text(json.dumps(result,indent=2))
    controls=dict(scenes=[v['name'] for v in output],candidates=[dict(id=c['id'],nm=c['wavelengths_nm'],weights=c.get('quadrature_weights_nm')) for c in candidates],metrics=output)
    page=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>光谱采样独立视觉复核</title>
<style>body{background:#141922;color:#dde5ee;font:16px system-ui;margin:24px}select,button{font:inherit;background:#273444;color:inherit;padding:8px;margin:4px;border:1px solid #576478}p{max-width:1100px;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}img{width:100%;display:block;image-rendering:auto;background:black}h3{margin:8px 0}#info{white-space:pre-line;font-family:monospace}a{color:#83c9ff}</style>
<h1>光谱采样候选：独立视图复核</h1><p>当前参考 LUT，CPU/f32。图中多重散射保持完整参考 RGB，只替换单次散射与地面边界的光谱积分。每个视图内曝光相同；不同视图自动曝光以看清细节。没有直射太阳盘。这不是压缩后多重散射或最终运行时效果。下拉选项可比较不同波长组合及积分权重。</p>
<select id="scene"></select><select id="candidate"></select><button id="blink">开始闪烁对比</button><button id="prev">上一候选</button><button id="next">下一候选</button>
<p id="info"></p><div class="grid"><section><h3>41 波长参考</h3><img id="ref"></section><section><h3>候选（闪烁时交替参考）</h3><img id="test"></section><section><h3>显示差分 ×20（黑色为零）</h3><img id="diff"></section><section><h3>线性 RGB 相对误差：黑 0%，橙 __HEAT_HALF__%，黄 ≥__HEAT_MAX__%</h3><img id="heat"></section></div>
<p><a id="sheet">打开全部候选联系表</a> · <a id="raw">下载线性 Rec.2020 浮点图（npz）</a> · <a href="visual.json">视觉指标</a> · <a href="../search.json">搜索记录</a></p>
<script>const DATA=__DATA__; const S=document.getElementById('scene'), C=document.getElementById('candidate');
for(const n of DATA.scenes) S.add(new Option(n,n)); for(const c of DATA.candidates) C.add(new Option(c.id+' | '+c.nm.join(', '),c.id));C.selectedIndex=2;
let timer=null,phase=false;function update(){let s=S.value,c=C.value;document.getElementById('ref').src=s+'_reference.png';document.getElementById('test').src=s+'_'+c+'.png';document.getElementById('diff').src=s+'_'+c+'_display_diff20.png';document.getElementById('heat').src=s+'_'+c+'_heat.png';document.getElementById('sheet').href=s+'_sheet.png';document.getElementById('raw').href=s+'_linear_rec2020.npz';let q=DATA.metrics.find(x=>x.name==s),m=q.candidates.find(x=>x.id==c),n=DATA.candidates.find(x=>x.id==c);document.getElementById('info').textContent='海拔 '+q.altitude_km+' km，太阳高度 '+q.sun_elevation_deg.toFixed(2)+'°，曝光 '+q.exposure.toFixed(4)+'\n'+n.nm.join(', ')+' nm\n'+(n.weights?'积分权重 '+n.weights.map(x=>x.toFixed(4)).join(', ')+' nm\n':'旧 demo 固定权重\n')+'P95 '+m.relative_error_percent.p95.toFixed(4)+'%，最大 '+m.relative_error_percent.max.toFixed(4)+'%，最大显示差 '+m.display_max_difference_8bit.toFixed(2)+'/255';phase=false;}
S.onchange=C.onchange=update;document.getElementById('prev').onclick=()=>{C.selectedIndex=(C.selectedIndex+C.length-1)%C.length;update()};document.getElementById('next').onclick=()=>{C.selectedIndex=(C.selectedIndex+1)%C.length;update()};document.getElementById('blink').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('blink').textContent='开始闪烁对比';update()}else{document.getElementById('blink').textContent='停止闪烁';timer=setInterval(()=>{phase=!phase;document.getElementById('test').src=S.value+(phase?'_reference':'_'+C.value)+'.png'},700)}};update();</script></html>'''
    page=page.replace('__HEAT_HALF__',f'{a.heat_max_percent/2:g}').replace('__HEAT_MAX__',f'{a.heat_max_percent:g}')
    (a.out/'index.html').write_text(page.replace('__DATA__',json.dumps(controls,ensure_ascii=False)),encoding='utf-8')
    print(json.dumps([{v['name']:{c['id']:c['relative_error_percent'] for c in v['candidates']}} for v in output],indent=2))


if __name__=='__main__':
    main()
