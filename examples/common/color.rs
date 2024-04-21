use num_traits::{AsPrimitive, Float};

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct SRGBColor<Ch> {
    pub red: Ch,
    pub green: Ch,
    pub blue: Ch,
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct RGBColor<Ch> {
    pub red: Ch,
    pub green: Ch,
    pub blue: Ch,
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct XYZColor<Ch> {
    pub x: Ch,
    pub y: Ch,
    pub z: Ch,
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct LABColor<Ch> {
    pub l: Ch,
    pub a: Ch,
    pub b: Ch,
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct LUVColor<Ch> {
    pub l: Ch,
    pub u: Ch,
    pub v: Ch,
}

fn channel_srgb_to_linear<Ch>(value: Ch) -> Ch
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    if value < 0.04045.as_() {
        value / 12.92.as_()
    } else {
        ((value + 0.055.as_()) / 1.055.as_()).powf(2.4.as_())
    }
}

fn channel_xyz_to_lab<Ch>(value: Ch) -> Ch
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    if value > (216.0 / 24389.0).as_() {
        value.cbrt()
    } else {
        value * 7.787.as_() + (16.0 / 116.0).as_()
    }
}

impl<Ch> SRGBColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    pub fn from_argb(argb: u32) -> Self {
        Self {
            red: (((argb & 0x00ff0000) >> 16) as f64 / 255.0).as_(),
            green: (((argb & 0x0000ff00) >> 8) as f64 / 255.0).as_(),
            blue: ((argb & 0x000000ff) as f64 / 255.0).as_(),
        }
    }
}

impl<Ch> From<SRGBColor<Ch>> for RGBColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: SRGBColor<Ch>) -> Self {
        Self {
            red: channel_srgb_to_linear(value.red),
            green: channel_srgb_to_linear(value.green),
            blue: channel_srgb_to_linear(value.blue),
        }
    }
}

impl<Ch> From<RGBColor<Ch>> for XYZColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: RGBColor<Ch>) -> Self {
        Self {
            x: value.red * 0.412424.as_()
                + value.green * 0.212656.as_()
                + value.blue * 0.0193324.as_(),
            y: value.red * 0.357579.as_()
                + value.green * 0.715158.as_()
                + value.blue * 0.119193.as_(),
            z: value.red * 0.180464.as_()
                + value.green * 0.0721856.as_()
                + value.blue * 0.950444.as_(),
        }
    }
}

impl<Ch> From<SRGBColor<Ch>> for XYZColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: SRGBColor<Ch>) -> Self {
        let value: RGBColor<Ch> = value.into();
        value.into()
    }
}

impl<Ch> From<XYZColor<Ch>> for LABColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: XYZColor<Ch>) -> Self {
        let x = channel_xyz_to_lab(value.x);
        let y = channel_xyz_to_lab(value.y);
        let z = channel_xyz_to_lab(value.z);
        Self {
            l: y * 116.0.as_() - 16.0.as_(),
            a: (x - y) * 500.0.as_(),
            b: (y - z) * 200.0.as_(),
        }
    }
}

impl<Ch> From<RGBColor<Ch>> for LABColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: RGBColor<Ch>) -> Self {
        let value: XYZColor<Ch> = value.into();
        value.into()
    }
}

impl<Ch> From<SRGBColor<Ch>> for LABColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: SRGBColor<Ch>) -> Self {
        let value: RGBColor<Ch> = value.into();
        let value: XYZColor<Ch> = value.into();
        value.into()
    }
}

impl<Ch> From<XYZColor<Ch>> for LUVColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: XYZColor<Ch>) -> Self {
        let denom = value.x + value.y * 15.0.as_() + value.z * 3.0.as_();
        let (u, v): (Ch, Ch) = if denom.is_zero() {
            (0.0.as_(), 0.0.as_())
        } else {
            (value.x * 4.0.as_() / denom, value.y * 9.0.as_() / denom)
        };
        let (d65_x, d65_y, d65_z) = (0.9481, 1.0, 1.073);
        let ref_u = ((4.0 * d65_x) / (d65_x + 15.0 * d65_y + 3.0 * d65_z)).as_();
        let ref_v = ((9.0 * d65_y) / (d65_x + 15.0 * d65_y + 3.0 * d65_z)).as_();
        let luminance = channel_xyz_to_lab(value.y) * 116.0.as_() - 16.0.as_();
        Self {
            l: luminance,
            u: luminance * 13.0.as_() * (u - ref_u),
            v: luminance * 13.0.as_() * (v - ref_v),
        }
    }
}

impl<Ch> From<RGBColor<Ch>> for LUVColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: RGBColor<Ch>) -> Self {
        let value: XYZColor<Ch> = value.into();
        value.into()
    }
}

impl<Ch> From<SRGBColor<Ch>> for LUVColor<Ch>
where
    Ch: 'static + Float,
    f64: AsPrimitive<Ch>,
{
    fn from(value: SRGBColor<Ch>) -> Self {
        let value: RGBColor<Ch> = value.into();
        let value: XYZColor<Ch> = value.into();
        value.into()
    }
}
