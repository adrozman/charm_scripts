      program processmrev
c** This reads the hubacr.dat file and outputs values in Report format
c** Unlike processhub, this supports MREV calculations taking an average.
      character*120 cmnt
      real hub(6),sumhub(6)
      open(99,file='fort.99',status='unknown')
      open(98,file='fort.98',access='append')
c**      write(*,500)
c** 500   format('Enter degs pitch: ',$)
c**      read(*,*) pitch
c**      pitch=0
      k = 0
      irev = 0
      sumhub = 0.0
10    read(99,1000,end=100) cmnt
1000  format(A)
      if (cmnt(4:12).ne.'RESULTANT') go to 10
c** only process every other time RESULTANT appears
c** because you only want the body frame results.
      if (k.eq.1) then
       k = 0
       go to 10
      end if
      k = 1
      irev = irev + 1
20    read(99,1000) cmnt
      if (cmnt(1:7).ne.'AVERAGE') go to 20
      backspace(99)
      read(99,1010) hub
1010  format(8X,6G12.0)
1020  format(7ES12.3E2)
c**      hub(1) = -hub(1)
c**      hub(3) = -hub(3)
c**      hub(4:6) = 12.0*hub(4:6)
      sumhub = sumhub + hub
      write(*,1020) hub
      go to 10
100   continue
      print*,'Total revs, trim + MREV = ',irev
      hub = sumhub/float(irev)
      write(98,1020) hub
      stop 
      end
