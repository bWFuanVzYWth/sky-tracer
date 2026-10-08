"""CPU tests for physical branches and the unchanged coordinate formula."""
import json
from pathlib import Path
import unittest
import numpy as np
import fit_sky_coordinates_cpu as fit

CALIBRATION=json.loads((Path(__file__).resolve().parents[2]/'crates/sky-realtime/configs/mapping_fit.json').read_text())

class Coordinates(unittest.TestCase):
    def test_monotone_cdf_and_inverse_inside_and_outside_sun(self):
        u=np.linspace(0,1,501)
        for h in [.002,.2,12,108,121,400]:
            for sun in [-.9,0,1.4]:
                for chart in range(3):
                    if h>=120 and chart==1:continue
                    p=fit.parameters(CALIBRATION,h,sun,chart)
                    e=fit.inverse(u,*p[:4])
                    self.assertTrue(np.all(np.diff(e)>0))
                    self.assertLess(float(np.max(np.abs(fit.cdf(e,*p[:4])-u))),2e-7)

    def test_space_last_cell_preserves_chord_and_exact_vacuum(self):
        h=400
        p=fit.parameters(CALIBRATION,h,-.1,0)
        node=fit.inverse(np.linspace(0,1,192),*p[:4])
        e=np.unique(np.concatenate([np.linspace(node[-2],node[-1],1025),node]))
        radius=6360+h
        d=np.maximum((radius*np.sin(e))**2-(h-120)*(radius+6480),0)
        rgb=np.sqrt(d)[:,None]*np.array([1,.4,.1])
        rgb[-1]=0
        curve=dict(altitude_km=h,sun_elevation_deg=np.rad2deg(-.1),chart=0,e=e,rgb=rgb)
        result=fit.predict(curve,CALIBRATION,256)
        tail=e>=node[-2]
        self.assertLess(float(np.max(np.abs(result[tail]-rgb[tail]))),1e-8)
        self.assertTrue(np.all(result[-1]==0))

    def test_exact_vacuum_does_not_create_relative_error_curvature(self):
        p=fit.parameters(CALIBRATION,400,-.1,0)
        e=fit.inverse(np.linspace(0,1,1025),*p[:4])
        curve=dict(altitude_km=400,sun_elevation_deg=np.rad2deg(-.1),chart=0,e=e,rgb=np.zeros((len(e),3)))
        values=fit.errors([curve],CALIBRATION,256,1)
        self.assertTrue(np.all(values==0))

if __name__=='__main__':unittest.main()
