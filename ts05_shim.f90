module ts05_shim
  use iso_c_binding, only: c_double, c_int
  implicit none
contains
  function t04_component(comp, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ps, x, y, z) result(b) bind(C)
    integer(c_int), value :: comp
    real(c_double), value :: p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
    real(c_double), value :: ps, x, y, z
    real(c_double) :: b
    real(c_double) :: parmod(10), bx, by, bz
    integer(c_int) :: iopt

    parmod = (/p1, p2, p3, p4, p5, p6, p7, p8, p9, p10/)
    iopt = 0_c_int
    call T04_s(iopt, parmod, ps, x, y, z, bx, by, bz)

    if (comp == 0_c_int) then
      b = bx
    else if (comp == 1_c_int) then
      b = by
    else
      b = bz
    end if
  end function t04_component

  subroutine t04_fill(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ps, x, y, z, out) bind(C)
    real(c_double), value :: p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
    real(c_double), value :: ps, x, y, z
    real(c_double) :: out(3)
    real(c_double) :: parmod(10), bx, by, bz
    integer(c_int) :: iopt

    parmod = (/p1, p2, p3, p4, p5, p6, p7, p8, p9, p10/)
    iopt = 0_c_int
    call T04_s(iopt, parmod, ps, x, y, z, bx, by, bz)
    out(1) = bx
    out(2) = by
    out(3) = bz
  end subroutine t04_fill
end module ts05_shim
