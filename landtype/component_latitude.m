% % SMAP
% SMF1(find(MSWEP_PA_new<250))=nan;
% SMF2(find(MSWEP_PA_new<250))=nan;
% SMF3(find(MSWEP_PA_new<250))=nan;
% SMF4(find(MSWEP_PA_new<250))=nan;
% [m,~]=size(SMF1);
% SMAP_Avg1(m,1)=0;
% count=1;
% for i=1:m
%     SMAP_Avg1(count,1)=nanmean(SMF1(i,:));
%     count=count+1;
% end
% SMAP_Avg2(m,1)=0;
% count=1;
% for i=1:m
%     SMAP_Avg2(count,1)=nanmean(SMF2(i,:));
%     count=count+1;
% end
% SMAP_Avg3(m,1)=0;
% count=1;
% for i=1:m
%     SMAP_Avg3(count,1)=nanmean(SMF3(i,:));
%     count=count+1;
% end
% SMAP_Avg4(m,1)=0;
% count=1;
% for i=1:m
%     SMAP_Avg4(count,1)=nanmean(SMF4(i,:));
%     count=count+1;
% end
% 
BCC_CSM_mrsosF1(find(MSWEP_PA_new<250))=nan;
BCC_CSM_mrsosF2(find(MSWEP_PA_new<250))=nan;
BCC_CSM_mrsosF3(find(MSWEP_PA_new<250))=nan;
BCC_CSM_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(BCC_CSM_mrsosF1);
BCC_CSM_mrsosF1(BCC_CSM_mrsosF1==BCC_CSM_mrsosF1(m,1))=nan;
BCC_CSM_mrsosF1(1:2,:)=nan;
BCC_CSM_mrsosF1(55:m,:)=nan;
BCC_CSM_avg1(m,1)=0;
count=1;
for i=1:m
    BCC_CSM_avg1(count,1)=nanmean(BCC_CSM_mrsosF1(i,:));
    count=count+1;
end
BCC_CSM_Avg1=flipud(BCC_CSM_avg1);
BCC_CSM_mrsosF2(BCC_CSM_mrsosF2==BCC_CSM_mrsosF2(m,1))=nan;
BCC_CSM_mrsosF2(1:2,:)=nan;
BCC_CSM_mrsosF2(55:m,:)=nan;
BCC_CSM_avg2(m,1)=0;
count=1;
for i=1:m
    BCC_CSM_avg2(count,1)=nanmean(BCC_CSM_mrsosF2(i,:));
    count=count+1;
end
BCC_CSM_Avg2=flipud(BCC_CSM_avg2);
BCC_CSM_mrsosF3(BCC_CSM_mrsosF3==BCC_CSM_mrsosF3(m,1))=nan;
BCC_CSM_mrsosF3(1:2,:)=nan;
BCC_CSM_mrsosF3(55:m,:)=nan;
BCC_CSM_avg3(m,1)=0;
count=1;
for i=1:m
    BCC_CSM_avg3(count,1)=nanmean(BCC_CSM_mrsosF3(i,:));
    count=count+1;
end
BCC_CSM_Avg3=flipud(BCC_CSM_avg3);
BCC_CSM_mrsosF4(BCC_CSM_mrsosF4==BCC_CSM_mrsosF4(m,1))=nan;
BCC_CSM_mrsosF4(1:2,:)=nan;
BCC_CSM_mrsosF4(55:m,:)=nan;
BCC_CSM_avg4(m,1)=0;
count=1;
for i=1:m
    BCC_CSM_avg4(count,1)=nanmean(BCC_CSM_mrsosF4(i,:));
    count=count+1;
end
BCC_CSM_Avg4=flipud(BCC_CSM_avg4);

BNU_ESM_mrsosF1(find(MSWEP_PA_new<250))=nan;
BNU_ESM_mrsosF2(find(MSWEP_PA_new<250))=nan;
BNU_ESM_mrsosF3(find(MSWEP_PA_new<250))=nan;
BNU_ESM_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(BNU_ESM_mrsosF1);
BNU_ESM_mrsosF1(BNU_ESM_mrsosF1==BNU_ESM_mrsosF1(m,1))=nan;
BNU_ESM_mrsosF1(1:2,:)=nan;
BNU_ESM_mrsosF1(55:m,:)=nan;
BNU_ESM_avg1(m,1)=0;
count=1;
for i=1:m
    BNU_ESM_avg1(count,1)=nanmean(BNU_ESM_mrsosF1(i,:));
    count=count+1;
end
BNU_ESM_Avg1=flipud(BNU_ESM_avg1);
BNU_ESM_mrsosF2(BNU_ESM_mrsosF2==BNU_ESM_mrsosF2(m,1))=nan;
BNU_ESM_mrsosF2(1:2,:)=nan;
BNU_ESM_mrsosF2(55:m,:)=nan;
BNU_ESM_avg2(m,1)=0;
count=1;
for i=1:m
    BNU_ESM_avg2(count,1)=nanmean(BNU_ESM_mrsosF2(i,:));
    count=count+1;
end
BNU_ESM_Avg2=flipud(BNU_ESM_avg2);
BNU_ESM_mrsosF3(BNU_ESM_mrsosF3==BNU_ESM_mrsosF3(m,1))=nan;
BNU_ESM_mrsosF3(1:2,:)=nan;
BNU_ESM_mrsosF3(55:m,:)=nan;
BNU_ESM_avg3(m,1)=0;
count=1;
for i=1:m
    BNU_ESM_avg3(count,1)=nanmean(BNU_ESM_mrsosF3(i,:));
    count=count+1;
end
BNU_ESM_Avg3=flipud(BNU_ESM_avg3);
BNU_ESM_mrsosF4(BNU_ESM_mrsosF4==BNU_ESM_mrsosF4(m,1))=nan;
BNU_ESM_mrsosF4(1:2,:)=nan;
BNU_ESM_mrsosF4(55:m,:)=nan;
BNU_ESM_avg4(m,1)=0;
count=1;
for i=1:m
    BNU_ESM_avg4(count,1)=nanmean(BNU_ESM_mrsosF4(i,:));
    count=count+1;
end
BNU_ESM_Avg4=flipud(BNU_ESM_avg4);

CanESM2_mrsosF1(find(MSWEP_PA_new<250))=nan;
CanESM2_mrsosF2(find(MSWEP_PA_new<250))=nan;
CanESM2_mrsosF3(find(MSWEP_PA_new<250))=nan;
CanESM2_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(CanESM2_mrsosF1);
CanESM2_mrsosF1(CanESM2_mrsosF1==CanESM2_mrsosF1(m,1))=nan;
CanESM2_mrsosF1(1:2,:)=nan;
CanESM2_mrsosF1(55:m,:)=nan;
CanESM2_avg1(m,1)=0;
count=1;
for i=1:m
    CanESM2_avg1(count,1)=nanmean(CanESM2_mrsosF1(i,:));
    count=count+1;
end
CanESM2_Avg1=flipud(CanESM2_avg1);
CanESM2_mrsosF2(CanESM2_mrsosF2==CanESM2_mrsosF2(m,1))=nan;
CanESM2_mrsosF2(1:2,:)=nan;
CanESM2_mrsosF2(55:m,:)=nan;
CanESM2_avg2(m,1)=0;
count=1;
for i=1:m
    CanESM2_avg2(count,1)=nanmean(CanESM2_mrsosF2(i,:));
    count=count+1;
end
CanESM2_Avg2=flipud(CanESM2_avg2);
CanESM2_mrsosF3(CanESM2_mrsosF3==CanESM2_mrsosF3(m,1))=nan;
CanESM2_mrsosF3(1:2,:)=nan;
CanESM2_mrsosF3(55:m,:)=nan;
CanESM2_avg3(m,1)=0;
count=1;
for i=1:m
    CanESM2_avg3(count,1)=nanmean(CanESM2_mrsosF3(i,:));
    count=count+1;
end
CanESM2_Avg3=flipud(CanESM2_avg3);
CanESM2_mrsosF4(CanESM2_mrsosF4==CanESM2_mrsosF4(m,1))=nan;
CanESM2_mrsosF4(1:2,:)=nan;
CanESM2_mrsosF4(55:m,:)=nan;
CanESM2_avg4(m,1)=0;
count=1;
for i=1:m
    CanESM2_avg4(count,1)=nanmean(CanESM2_mrsosF4(i,:));
    count=count+1;
end
CanESM2_Avg4=flipud(CanESM2_avg4);

CNRM_CM5_mrsosF1(find(MSWEP_PA_new<250))=nan;
CNRM_CM5_mrsosF2(find(MSWEP_PA_new<250))=nan;
CNRM_CM5_mrsosF3(find(MSWEP_PA_new<250))=nan;
CNRM_CM5_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(CNRM_CM5_mrsosF1);
CNRM_CM5_mrsosF1(CNRM_CM5_mrsosF1==CNRM_CM5_mrsosF1(m,1))=nan;
CNRM_CM5_mrsosF1(1:4,:)=nan;
CNRM_CM5_mrsosF1(108:m,:)=nan;
CNRM_CM5_avg1(m,1)=0;
count=1;
for i=1:m
    CNRM_CM5_avg1(count,1)=nanmean(CNRM_CM5_mrsosF1(i,:));
    count=count+1;
end
CNRM_CM5_Avg1=flipud(CNRM_CM5_avg1);
CNRM_CM5_mrsosF2(CNRM_CM5_mrsosF2==CNRM_CM5_mrsosF2(m,1))=nan;
CNRM_CM5_mrsosF2(1:4,:)=nan;
CNRM_CM5_mrsosF2(108:m,:)=nan;
CNRM_CM5_avg2(m,1)=0;
count=1;
for i=1:m
    CNRM_CM5_avg2(count,1)=nanmean(CNRM_CM5_mrsosF2(i,:));
    count=count+1;
end
CNRM_CM5_Avg2=flipud(CNRM_CM5_avg2);
CNRM_CM5_mrsosF3(CNRM_CM5_mrsosF3==CNRM_CM5_mrsosF3(m,1))=nan;
CNRM_CM5_mrsosF3(1:4,:)=nan;
CNRM_CM5_mrsosF3(108:m,:)=nan;
CNRM_CM5_avg3(m,1)=0;
count=1;
for i=1:m
    CNRM_CM5_avg3(count,1)=nanmean(CNRM_CM5_mrsosF3(i,:));
    count=count+1;
end
CNRM_CM5_Avg3=flipud(CNRM_CM5_avg3);
CNRM_CM5_mrsosF4(CNRM_CM5_mrsosF4==CNRM_CM5_mrsosF4(m,1))=nan;
CNRM_CM5_mrsosF4(1:4,:)=nan;
CNRM_CM5_mrsosF4(108:m,:)=nan;
CNRM_CM5_avg4(m,1)=0;
count=1;
for i=1:m
    CNRM_CM5_avg4(count,1)=nanmean(CNRM_CM5_mrsosF4(i,:));
    count=count+1;
end
CNRM_CM5_Avg4=flipud(CNRM_CM5_avg4);

CSIRO_Mk_mrsosF1(find(MSWEP_PA_new<250))=nan;
CSIRO_Mk_mrsosF2(find(MSWEP_PA_new<250))=nan;
CSIRO_Mk_mrsosF3(find(MSWEP_PA_new<250))=nan;
CSIRO_Mk_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(CSIRO_Mk_mrsosF1);
CSIRO_Mk_mrsosF1(CSIRO_Mk_mrsosF1==CSIRO_Mk_mrsosF1(m,1))=nan;
CSIRO_Mk_mrsosF1(1:3,:)=nan;
CSIRO_Mk_mrsosF1(81:m,:)=nan;
CSIRO_Mk_avg1(m,1)=0;
count=1;
for i=1:m
    CSIRO_Mk_avg1(count,1)=nanmean(CSIRO_Mk_mrsosF1(i,:));
    count=count+1;
end
CSIRO_Mk_Avg1=flipud(CSIRO_Mk_avg1);
CSIRO_Mk_mrsosF2(CSIRO_Mk_mrsosF2==CSIRO_Mk_mrsosF2(m,1))=nan;
CSIRO_Mk_mrsosF2(1:3,:)=nan;
CSIRO_Mk_mrsosF2(81:m,:)=nan;
CSIRO_Mk_avg2(m,1)=0;
count=1;
for i=1:m
    CSIRO_Mk_avg2(count,1)=nanmean(CSIRO_Mk_mrsosF2(i,:));
    count=count+1;
end
CSIRO_Mk_Avg2=flipud(CSIRO_Mk_avg2);
CSIRO_Mk_mrsosF3(CSIRO_Mk_mrsosF3==CSIRO_Mk_mrsosF3(m,1))=nan;
CSIRO_Mk_mrsosF3(1:3,:)=nan;
CSIRO_Mk_mrsosF3(81:m,:)=nan;
CSIRO_Mk_avg3(m,1)=0;
count=1;
for i=1:m
    CSIRO_Mk_avg3(count,1)=nanmean(CSIRO_Mk_mrsosF3(i,:));
    count=count+1;
end
CSIRO_Mk_Avg3=flipud(CSIRO_Mk_avg3);
CSIRO_Mk_mrsosF4(CSIRO_Mk_mrsosF4==CSIRO_Mk_mrsosF4(m,1))=nan;
CSIRO_Mk_mrsosF4(1:3,:)=nan;
CSIRO_Mk_mrsosF4(81:m,:)=nan;
CSIRO_Mk_avg4(m,1)=0;
count=1;
for i=1:m
    CSIRO_Mk_avg4(count,1)=nanmean(CSIRO_Mk_mrsosF4(i,:));
    count=count+1;
end
CSIRO_Mk_Avg4=flipud(CSIRO_Mk_avg4);

GFDL_CM3_mrsosF1(find(MSWEP_PA_new<250))=nan;
GFDL_CM3_mrsosF2(find(MSWEP_PA_new<250))=nan;
GFDL_CM3_mrsosF3(find(MSWEP_PA_new<250))=nan;
GFDL_CM3_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(GFDL_CM3_mrsosF1);
GFDL_CM3_mrsosF1(GFDL_CM3_mrsosF1==GFDL_CM3_mrsosF1(m,1))=nan;
GFDL_CM3_mrsosF1(1:3,:)=nan;
GFDL_CM3_mrsosF1(76:m,:)=nan;
GFDL_CM3_avg1(m,1)=0;
count=1;
for i=1:m
    GFDL_CM3_avg1(count,1)=nanmean(GFDL_CM3_mrsosF1(i,:));
    count=count+1;
end
GFDL_CM3_Avg1=flipud(GFDL_CM3_avg1);
GFDL_CM3_mrsosF2(GFDL_CM3_mrsosF2==GFDL_CM3_mrsosF2(m,1))=nan;
GFDL_CM3_mrsosF2(1:3,:)=nan;
GFDL_CM3_mrsosF2(76:m,:)=nan;
GFDL_CM3_avg2(m,1)=0;
count=1;
for i=1:m
    GFDL_CM3_avg2(count,1)=nanmean(GFDL_CM3_mrsosF2(i,:));
    count=count+1;
end
GFDL_CM3_Avg2=flipud(GFDL_CM3_avg2);
GFDL_CM3_mrsosF3(GFDL_CM3_mrsosF3==GFDL_CM3_mrsosF3(m,1))=nan;
GFDL_CM3_mrsosF3(1:3,:)=nan;
GFDL_CM3_mrsosF3(76:m,:)=nan;
GFDL_CM3_avg3(m,1)=0;
count=1;
for i=1:m
    GFDL_CM3_avg3(count,1)=nanmean(GFDL_CM3_mrsosF3(i,:));
    count=count+1;
end
GFDL_CM3_Avg3=flipud(GFDL_CM3_avg3);
GFDL_CM3_mrsosF4(GFDL_CM3_mrsosF4==GFDL_CM3_mrsosF4(m,1))=nan;
GFDL_CM3_mrsosF4(1:3,:)=nan;
GFDL_CM3_mrsosF4(76:m,:)=nan;
GFDL_CM3_avg4(m,1)=0;
count=1;
for i=1:m
    GFDL_CM3_avg4(count,1)=nanmean(GFDL_CM3_mrsosF4(i,:));
    count=count+1;
end
GFDL_CM3_Avg4=flipud(GFDL_CM3_avg4);

GFDL_ESM2G_mrsosF1(find(MSWEP_PA_new<250))=nan;
GFDL_ESM2G_mrsosF2(find(MSWEP_PA_new<250))=nan;
GFDL_ESM2G_mrsosF3(find(MSWEP_PA_new<250))=nan;
GFDL_ESM2G_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(GFDL_ESM2G_mrsosF1);
GFDL_ESM2G_mrsosF1(GFDL_ESM2G_mrsosF1==GFDL_ESM2G_mrsosF1(m,1))=nan;
GFDL_ESM2G_mrsosF1(1:3,:)=nan;
GFDL_ESM2G_mrsosF1(76:m,:)=nan;
GFDL_ESM2G_avg1(m,1)=0;
count=1;
for i=1:m
    GFDL_ESM2G_avg1(count,1)=nanmean(GFDL_ESM2G_mrsosF1(i,:));
    count=count+1;
end
GFDL_ESM2G_Avg1=flipud(GFDL_ESM2G_avg1);
GFDL_ESM2G_mrsosF2(GFDL_ESM2G_mrsosF2==GFDL_ESM2G_mrsosF2(m,1))=nan;
GFDL_ESM2G_mrsosF2(1:3,:)=nan;
GFDL_ESM2G_mrsosF2(76:m,:)=nan;
GFDL_ESM2G_avg2(m,1)=0;
count=1;
for i=1:m
    GFDL_ESM2G_avg2(count,1)=nanmean(GFDL_ESM2G_mrsosF2(i,:));
    count=count+1;
end
GFDL_ESM2G_Avg2=flipud(GFDL_ESM2G_avg2);
GFDL_ESM2G_mrsosF3(GFDL_ESM2G_mrsosF3==GFDL_ESM2G_mrsosF3(m,1))=nan;
GFDL_ESM2G_mrsosF3(1:3,:)=nan;
GFDL_ESM2G_mrsosF3(76:m,:)=nan;
GFDL_ESM2G_avg3(m,1)=0;
count=1;
for i=1:m
    GFDL_ESM2G_avg3(count,1)=nanmean(GFDL_ESM2G_mrsosF3(i,:));
    count=count+1;
end
GFDL_ESM2G_Avg3=flipud(GFDL_ESM2G_avg3);
GFDL_ESM2G_mrsosF4(GFDL_ESM2G_mrsosF4==GFDL_ESM2G_mrsosF4(m,1))=nan;
GFDL_ESM2G_mrsosF4(1:3,:)=nan;
GFDL_ESM2G_mrsosF4(76:m,:)=nan;
GFDL_ESM2G_avg4(m,1)=0;
count=1;
for i=1:m
    GFDL_ESM2G_avg4(count,1)=nanmean(GFDL_ESM2G_mrsosF4(i,:));
    count=count+1;
end
GFDL_ESM2G_Avg4=flipud(GFDL_ESM2G_avg4);

GFDL_ESM2M_mrsosF1(find(MSWEP_PA_new<250))=nan;
GFDL_ESM2M_mrsosF2(find(MSWEP_PA_new<250))=nan;
GFDL_ESM2M_mrsosF3(find(MSWEP_PA_new<250))=nan;
GFDL_ESM2M_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(GFDL_ESM2M_mrsosF1);
GFDL_ESM2M_mrsosF1(GFDL_ESM2M_mrsosF1==GFDL_ESM2M_mrsosF1(m,1))=nan;
GFDL_ESM2M_mrsosF1(1:3,:)=nan;
GFDL_ESM2M_mrsosF1(76:m,:)=nan;
GFDL_ESM2M_avg1(m,1)=0;
count=1;
for i=1:m
    GFDL_ESM2M_avg1(count,1)=nanmean(GFDL_ESM2M_mrsosF1(i,:));
    count=count+1;
end
GFDL_ESM2M_Avg1=flipud(GFDL_ESM2M_avg1);
GFDL_ESM2M_mrsosF2(GFDL_ESM2M_mrsosF2==GFDL_ESM2M_mrsosF2(m,1))=nan;
GFDL_ESM2M_mrsosF2(1:3,:)=nan;
GFDL_ESM2M_mrsosF2(76:m,:)=nan;
GFDL_ESM2M_avg2(m,1)=0;
count=1;
for i=1:m
    GFDL_ESM2M_avg2(count,1)=nanmean(GFDL_ESM2M_mrsosF2(i,:));
    count=count+1;
end
GFDL_ESM2M_Avg2=flipud(GFDL_ESM2M_avg2);
GFDL_ESM2M_mrsosF3(GFDL_ESM2M_mrsosF3==GFDL_ESM2M_mrsosF3(m,1))=nan;
GFDL_ESM2M_mrsosF3(1:3,:)=nan;
GFDL_ESM2M_mrsosF3(76:m,:)=nan;
GFDL_ESM2M_avg3(m,1)=0;
count=1;
for i=1:m
    GFDL_ESM2M_avg3(count,1)=nanmean(GFDL_ESM2M_mrsosF3(i,:));
    count=count+1;
end
GFDL_ESM2M_Avg3=flipud(GFDL_ESM2M_avg3);
GFDL_ESM2M_mrsosF4(GFDL_ESM2M_mrsosF4==GFDL_ESM2M_mrsosF4(m,1))=nan;
GFDL_ESM2M_mrsosF4(1:3,:)=nan;
GFDL_ESM2M_mrsosF4(76:m,:)=nan;
GFDL_ESM2M_avg4(m,1)=0;
count=1;
for i=1:m
    GFDL_ESM2M_avg4(count,1)=nanmean(GFDL_ESM2M_mrsosF4(i,:));
    count=count+1;
end
GFDL_ESM2M_Avg4=flipud(GFDL_ESM2M_avg4);

HadGEM2_CC_mrsosF1(find(MSWEP_PA_new<250))=nan;
HadGEM2_CC_mrsosF2(find(MSWEP_PA_new<250))=nan;
HadGEM2_CC_mrsosF3(find(MSWEP_PA_new<250))=nan;
HadGEM2_CC_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(HadGEM2_CC_mrsosF1);
HadGEM2_CC_mrsosF1(HadGEM2_CC_mrsosF1==HadGEM2_CC_mrsosF1(m,1))=nan;
HadGEM2_CC_mrsosF1(1:5,:)=nan;
HadGEM2_CC_mrsosF1(121:m,:)=nan;
HadGEM2_CC_avg1(m,1)=0;
count=1;
for i=1:m
    HadGEM2_CC_avg1(count,1)=nanmean(HadGEM2_CC_mrsosF1(i,:));
    count=count+1;
end
HadGEM2_CC_Avg1=flipud(HadGEM2_CC_avg1);
HadGEM2_CC_mrsosF2(HadGEM2_CC_mrsosF2==HadGEM2_CC_mrsosF2(m,1))=nan;
HadGEM2_CC_mrsosF2(1:5,:)=nan;
HadGEM2_CC_mrsosF2(121:m,:)=nan;
HadGEM2_CC_avg2(m,1)=0;
count=1;
for i=1:m
    HadGEM2_CC_avg2(count,1)=nanmean(HadGEM2_CC_mrsosF2(i,:));
    count=count+1;
end
HadGEM2_CC_Avg2=flipud(HadGEM2_CC_avg2);
HadGEM2_CC_mrsosF3(HadGEM2_CC_mrsosF3==HadGEM2_CC_mrsosF3(m,1))=nan;
HadGEM2_CC_mrsosF3(1:5,:)=nan;
HadGEM2_CC_mrsosF3(121:m,:)=nan;
HadGEM2_CC_avg3(m,1)=0;
count=1;
for i=1:m
    HadGEM2_CC_avg3(count,1)=nanmean(HadGEM2_CC_mrsosF3(i,:));
    count=count+1;
end
HadGEM2_CC_Avg3=flipud(HadGEM2_CC_avg3);
HadGEM2_CC_mrsosF4(HadGEM2_CC_mrsosF4==HadGEM2_CC_mrsosF4(m,1))=nan;
HadGEM2_CC_mrsosF4(1:5,:)=nan;
HadGEM2_CC_mrsosF4(121:m,:)=nan;
HadGEM2_CC_avg4(m,1)=0;
count=1;
for i=1:m
    HadGEM2_CC_avg4(count,1)=nanmean(HadGEM2_CC_mrsosF4(i,:));
    count=count+1;
end
HadGEM2_CC_Avg4=flipud(HadGEM2_CC_avg4);

HadGEM2_ES_mrsosF1(find(MSWEP_PA_new<250))=nan;
HadGEM2_ES_mrsosF2(find(MSWEP_PA_new<250))=nan;
HadGEM2_ES_mrsosF3(find(MSWEP_PA_new<250))=nan;
HadGEM2_ES_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(HadGEM2_ES_mrsosF1);
HadGEM2_ES_mrsosF1(HadGEM2_ES_mrsosF1==HadGEM2_ES_mrsosF1(m,1))=nan;
HadGEM2_ES_mrsosF1(1:5,:)=nan;
HadGEM2_ES_mrsosF1(121:m,:)=nan;
HadGEM2_ES_avg1(m,1)=0;
count=1;
for i=1:m
    HadGEM2_ES_avg1(count,1)=nanmean(HadGEM2_ES_mrsosF1(i,:));
    count=count+1;
end
HadGEM2_ES_Avg1=flipud(HadGEM2_ES_avg1);
HadGEM2_ES_mrsosF2(HadGEM2_ES_mrsosF2==HadGEM2_ES_mrsosF2(m,1))=nan;
HadGEM2_ES_mrsosF2(1:5,:)=nan;
HadGEM2_ES_mrsosF2(121:m,:)=nan;
HadGEM2_ES_avg2(m,1)=0;
count=1;
for i=1:m
    HadGEM2_ES_avg2(count,1)=nanmean(HadGEM2_ES_mrsosF2(i,:));
    count=count+1;
end
HadGEM2_ES_Avg2=flipud(HadGEM2_ES_avg2);
HadGEM2_ES_mrsosF3(HadGEM2_ES_mrsosF3==HadGEM2_ES_mrsosF3(m,1))=nan;
HadGEM2_ES_mrsosF3(1:5,:)=nan;
HadGEM2_ES_mrsosF3(121:m,:)=nan;
HadGEM2_ES_avg3(m,1)=0;
count=1;
for i=1:m
    HadGEM2_ES_avg3(count,1)=nanmean(HadGEM2_ES_mrsosF3(i,:));
    count=count+1;
end
HadGEM2_ES_Avg3=flipud(HadGEM2_ES_avg3);
HadGEM2_ES_mrsosF4(HadGEM2_ES_mrsosF4==HadGEM2_ES_mrsosF4(m,1))=nan;
HadGEM2_ES_mrsosF4(1:5,:)=nan;
HadGEM2_ES_mrsosF4(121:m,:)=nan;
HadGEM2_ES_avg4(m,1)=0;
count=1;
for i=1:m
    HadGEM2_ES_avg4(count,1)=nanmean(HadGEM2_ES_mrsosF4(i,:));
    count=count+1;
end
HadGEM2_ES_Avg4=flipud(HadGEM2_ES_avg4);

inmcm4_mrsosF1(find(MSWEP_PA_new<250))=nan;
inmcm4_mrsosF2(find(MSWEP_PA_new<250))=nan;
inmcm4_mrsosF3(find(MSWEP_PA_new<250))=nan;
inmcm4_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(inmcm4_mrsosF1);
inmcm4_mrsosF1(inmcm4_mrsosF1==inmcm4_mrsosF1(m,1))=nan;
inmcm4_mrsosF1(1:3,:)=nan;
inmcm4_mrsosF1(101:m,:)=nan;
inmcm4_avg1(m,1)=0;
count=1;
for i=1:m
    inmcm4_avg1(count,1)=nanmean(inmcm4_mrsosF1(i,:));
    count=count+1;
end
inmcm4_Avg1=flipud(inmcm4_avg1);
inmcm4_mrsosF2(inmcm4_mrsosF2==inmcm4_mrsosF2(m,1))=nan;
inmcm4_mrsosF2(1:3,:)=nan;
inmcm4_mrsosF2(101:m,:)=nan;
inmcm4_avg2(m,1)=0;
count=1;
for i=1:m
    inmcm4_avg2(count,1)=nanmean(inmcm4_mrsosF2(i,:));
    count=count+1;
end
inmcm4_Avg2=flipud(inmcm4_avg2);
inmcm4_mrsosF3(inmcm4_mrsosF3==inmcm4_mrsosF3(m,1))=nan;
inmcm4_mrsosF3(1:3,:)=nan;
inmcm4_mrsosF3(101:m,:)=nan;
inmcm4_avg3(m,1)=0;
count=1;
for i=1:m
    inmcm4_avg3(count,1)=nanmean(inmcm4_mrsosF3(i,:));
    count=count+1;
end
inmcm4_Avg3=flipud(inmcm4_avg3);
inmcm4_mrsosF4(inmcm4_mrsosF4==inmcm4_mrsosF4(m,1))=nan;
inmcm4_mrsosF4(1:3,:)=nan;
inmcm4_mrsosF4(101:m,:)=nan;
inmcm4_avg4(m,1)=0;
count=1;
for i=1:m
    inmcm4_avg4(count,1)=nanmean(inmcm4_mrsosF4(i,:));
    count=count+1;
end
inmcm4_Avg4=flipud(inmcm4_avg4);

MIROC5_mrsosF1(find(MSWEP_PA_new<250))=nan;
MIROC5_mrsosF2(find(MSWEP_PA_new<250))=nan;
MIROC5_mrsosF3(find(MSWEP_PA_new<250))=nan;
MIROC5_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(MIROC5_mrsosF1);
MIROC5_mrsosF1(MIROC5_mrsosF1==MIROC5_mrsosF1(m,1))=nan;
MIROC5_mrsosF1(1:4,:)=nan;
MIROC5_mrsosF1(108:m,:)=nan;
MIROC5_avg1(m,1)=0;
count=1;
for i=1:m
    MIROC5_avg1(count,1)=nanmean(MIROC5_mrsosF1(i,:));
    count=count+1;
end
MIROC5_Avg1=flipud(MIROC5_avg1);
MIROC5_mrsosF2(MIROC5_mrsosF2==MIROC5_mrsosF2(m,1))=nan;
MIROC5_mrsosF2(1:4,:)=nan;
MIROC5_mrsosF2(108:m,:)=nan;
MIROC5_avg2(m,1)=0;
count=1;
for i=1:m
    MIROC5_avg2(count,1)=nanmean(MIROC5_mrsosF2(i,:));
    count=count+1;
end
MIROC5_Avg2=flipud(MIROC5_avg2);
MIROC5_mrsosF3(MIROC5_mrsosF3==MIROC5_mrsosF3(m,1))=nan;
MIROC5_mrsosF3(1:4,:)=nan;
MIROC5_mrsosF3(108:m,:)=nan;
MIROC5_avg3(m,1)=0;
count=1;
for i=1:m
    MIROC5_avg3(count,1)=nanmean(MIROC5_mrsosF3(i,:));
    count=count+1;
end
MIROC5_Avg3=flipud(MIROC5_avg3);
MIROC5_mrsosF4(MIROC5_mrsosF4==MIROC5_mrsosF4(m,1))=nan;
MIROC5_mrsosF4(1:4,:)=nan;
MIROC5_mrsosF4(108:m,:)=nan;
MIROC5_avg4(m,1)=0;
count=1;
for i=1:m
    MIROC5_avg4(count,1)=nanmean(MIROC5_mrsosF4(i,:));
    count=count+1;
end
MIROC5_Avg4=flipud(MIROC5_avg4);

MIROC_ESM_mrsosF1(find(MSWEP_PA_new<250))=nan;
MIROC_ESM_mrsosF2(find(MSWEP_PA_new<250))=nan;
MIROC_ESM_mrsosF3(find(MSWEP_PA_new<250))=nan;
MIROC_ESM_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(MIROC_ESM_mrsosF1);
MIROC_ESM_mrsosF1(MIROC_ESM_mrsosF1==MIROC_ESM_mrsosF1(m,1))=nan;
MIROC_ESM_mrsosF1(1:2,:)=nan;
MIROC_ESM_mrsosF1(55:m,:)=nan;
MIROC_ESM_avg1(m,1)=0;
count=1;
for i=1:m
    MIROC_ESM_avg1(count,1)=nanmean(MIROC_ESM_mrsosF1(i,:));
    count=count+1;
end
MIROC_ESM_Avg1=flipud(MIROC_ESM_avg1);
MIROC_ESM_mrsosF2(MIROC_ESM_mrsosF2==MIROC_ESM_mrsosF2(m,1))=nan;
MIROC_ESM_mrsosF2(1:2,:)=nan;
MIROC_ESM_mrsosF2(55:m,:)=nan;
MIROC_ESM_avg2(m,1)=0;
count=1;
for i=1:m
    MIROC_ESM_avg2(count,1)=nanmean(MIROC_ESM_mrsosF2(i,:));
    count=count+1;
end
MIROC_ESM_Avg2=flipud(MIROC_ESM_avg2);
MIROC_ESM_mrsosF3(MIROC_ESM_mrsosF3==MIROC_ESM_mrsosF3(m,1))=nan;
MIROC_ESM_mrsosF3(1:2,:)=nan;
MIROC_ESM_mrsosF3(55:m,:)=nan;
MIROC_ESM_avg3(m,1)=0;
count=1;
for i=1:m
    MIROC_ESM_avg3(count,1)=nanmean(MIROC_ESM_mrsosF3(i,:));
    count=count+1;
end
MIROC_ESM_Avg3=flipud(MIROC_ESM_avg3);
MIROC_ESM_mrsosF4(MIROC_ESM_mrsosF4==MIROC_ESM_mrsosF4(m,1))=nan;
MIROC_ESM_mrsosF4(1:2,:)=nan;
MIROC_ESM_mrsosF4(55:m,:)=nan;
MIROC_ESM_avg4(m,1)=0;
count=1;
for i=1:m
    MIROC_ESM_avg4(count,1)=nanmean(MIROC_ESM_mrsosF4(i,:));
    count=count+1;
end
MIROC_ESM_Avg4=flipud(MIROC_ESM_avg4);

MIROC_ESM_CHEM_mrsosF1(find(MSWEP_PA_new<250))=nan;
MIROC_ESM_CHEM_mrsosF2(find(MSWEP_PA_new<250))=nan;
MIROC_ESM_CHEM_mrsosF3(find(MSWEP_PA_new<250))=nan;
MIROC_ESM_CHEM_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(MIROC_ESM_CHEM_mrsosF1);
MIROC_ESM_CHEM_mrsosF1(MIROC_ESM_CHEM_mrsosF1==MIROC_ESM_CHEM_mrsosF1(m,1))=nan;
MIROC_ESM_CHEM_mrsosF1(1:2,:)=nan;
MIROC_ESM_CHEM_mrsosF1(55:m,:)=nan;
MIROC_ESM_CHEM_avg1(m,1)=0;
count=1;
for i=1:m
    MIROC_ESM_CHEM_avg1(count,1)=nanmean(MIROC_ESM_CHEM_mrsosF1(i,:));
    count=count+1;
end
MIROC_ESM_CHEM_Avg1=flipud(MIROC_ESM_CHEM_avg1);
MIROC_ESM_CHEM_mrsosF2(MIROC_ESM_CHEM_mrsosF2==MIROC_ESM_CHEM_mrsosF2(m,1))=nan;
MIROC_ESM_CHEM_mrsosF2(1:2,:)=nan;
MIROC_ESM_CHEM_mrsosF2(55:m,:)=nan;
MIROC_ESM_CHEM_avg2(m,1)=0;
count=1;
for i=1:m
    MIROC_ESM_CHEM_avg2(count,1)=nanmean(MIROC_ESM_CHEM_mrsosF2(i,:));
    count=count+1;
end
MIROC_ESM_CHEM_Avg2=flipud(MIROC_ESM_CHEM_avg2);
MIROC_ESM_CHEM_mrsosF3(MIROC_ESM_CHEM_mrsosF3==MIROC_ESM_CHEM_mrsosF3(m,1))=nan;
MIROC_ESM_CHEM_mrsosF3(1:2,:)=nan;
MIROC_ESM_CHEM_mrsosF3(55:m,:)=nan;
MIROC_ESM_CHEM_avg3(m,1)=0;
count=1;
for i=1:m
    MIROC_ESM_CHEM_avg3(count,1)=nanmean(MIROC_ESM_CHEM_mrsosF3(i,:));
    count=count+1;
end
MIROC_ESM_CHEM_Avg3=flipud(MIROC_ESM_CHEM_avg3);
MIROC_ESM_CHEM_mrsosF4(MIROC_ESM_CHEM_mrsosF4==MIROC_ESM_CHEM_mrsosF4(m,1))=nan;
MIROC_ESM_CHEM_mrsosF4(1:2,:)=nan;
MIROC_ESM_CHEM_mrsosF4(55:m,:)=nan;
MIROC_ESM_CHEM_avg4(m,1)=0;
count=1;
for i=1:m
    MIROC_ESM_CHEM_avg4(count,1)=nanmean(MIROC_ESM_CHEM_mrsosF4(i,:));
    count=count+1;
end
MIROC_ESM_CHEM_Avg4=flipud(MIROC_ESM_CHEM_avg4);

MRI_CGCM3_mrsosF1(find(MSWEP_PA_new<250))=nan;
MRI_CGCM3_mrsosF2(find(MSWEP_PA_new<250))=nan;
MRI_CGCM3_mrsosF3(find(MSWEP_PA_new<250))=nan;
MRI_CGCM3_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(MRI_CGCM3_mrsosF1);
MRI_CGCM3_mrsosF1(MRI_CGCM3_mrsosF1==MRI_CGCM3_mrsosF1(m,1))=nan;
MRI_CGCM3_mrsosF1(1:5,:)=nan;
MRI_CGCM3_mrsosF1(135:m,:)=nan;
MRI_CGCM3_avg1(m,1)=0;
count=1;
for i=1:m
    MRI_CGCM3_avg1(count,1)=nanmean(MRI_CGCM3_mrsosF1(i,:));
    count=count+1;
end
MRI_CGCM3_Avg1=flipud(MRI_CGCM3_avg1);
MRI_CGCM3_mrsosF2(MRI_CGCM3_mrsosF2==MRI_CGCM3_mrsosF2(m,1))=nan;
MRI_CGCM3_mrsosF2(1:5,:)=nan;
MRI_CGCM3_mrsosF2(135:m,:)=nan;
MRI_CGCM3_avg2(m,1)=0;
count=1;
for i=1:m
    MRI_CGCM3_avg2(count,1)=nanmean(MRI_CGCM3_mrsosF2(i,:));
    count=count+1;
end
MRI_CGCM3_Avg2=flipud(MRI_CGCM3_avg2);
MRI_CGCM3_mrsosF3(MRI_CGCM3_mrsosF3==MRI_CGCM3_mrsosF3(m,1))=nan;
MRI_CGCM3_mrsosF3(1:5,:)=nan;
MRI_CGCM3_mrsosF3(135:m,:)=nan;
MRI_CGCM3_avg3(m,1)=0;
count=1;
for i=1:m
    MRI_CGCM3_avg3(count,1)=nanmean(MRI_CGCM3_mrsosF3(i,:));
    count=count+1;
end
MRI_CGCM3_Avg3=flipud(MRI_CGCM3_avg3);
MRI_CGCM3_mrsosF4(MRI_CGCM3_mrsosF4==MRI_CGCM3_mrsosF4(m,1))=nan;
MRI_CGCM3_mrsosF4(1:5,:)=nan;
MRI_CGCM3_mrsosF4(135:m,:)=nan;
MRI_CGCM3_avg4(m,1)=0;
count=1;
for i=1:m
    MRI_CGCM3_avg4(count,1)=nanmean(MRI_CGCM3_mrsosF4(i,:));
    count=count+1;
end
MRI_CGCM3_Avg4=flipud(MRI_CGCM3_avg4);

MRI_ESM1_mrsosF1(find(MSWEP_PA_new<250))=nan;
MRI_ESM1_mrsosF2(find(MSWEP_PA_new<250))=nan;
MRI_ESM1_mrsosF3(find(MSWEP_PA_new<250))=nan;
MRI_ESM1_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(MRI_ESM1_mrsosF1);
MRI_ESM1_mrsosF1(MRI_ESM1_mrsosF1==MRI_ESM1_mrsosF1(m,1))=nan;
MRI_ESM1_mrsosF1(1:5,:)=nan;
MRI_ESM1_mrsosF1(135:m,:)=nan;
MRI_ESM1_avg1(m,1)=0;
count=1;
for i=1:m
    MRI_ESM1_avg1(count,1)=nanmean(MRI_ESM1_mrsosF1(i,:));
    count=count+1;
end
MRI_ESM1_Avg1=flipud(MRI_ESM1_avg1);
MRI_ESM1_mrsosF2(MRI_ESM1_mrsosF2==MRI_ESM1_mrsosF2(m,1))=nan;
MRI_ESM1_mrsosF2(1:5,:)=nan;
MRI_ESM1_mrsosF2(135:m,:)=nan;
MRI_ESM1_avg2(m,1)=0;
count=1;
for i=1:m
    MRI_ESM1_avg2(count,1)=nanmean(MRI_ESM1_mrsosF2(i,:));
    count=count+1;
end
MRI_ESM1_Avg2=flipud(MRI_ESM1_avg2);
MRI_ESM1_mrsosF3(MRI_ESM1_mrsosF3==MRI_ESM1_mrsosF3(m,1))=nan;
MRI_ESM1_mrsosF3(1:5,:)=nan;
MRI_ESM1_mrsosF3(135:m,:)=nan;
MRI_ESM1_avg3(m,1)=0;
count=1;
for i=1:m
    MRI_ESM1_avg3(count,1)=nanmean(MRI_ESM1_mrsosF3(i,:));
    count=count+1;
end
MRI_ESM1_Avg3=flipud(MRI_ESM1_avg3);
MRI_ESM1_mrsosF4(MRI_ESM1_mrsosF4==MRI_ESM1_mrsosF4(m,1))=nan;
MRI_ESM1_mrsosF4(1:5,:)=nan;
MRI_ESM1_mrsosF4(135:m,:)=nan;
MRI_ESM1_avg4(m,1)=0;
count=1;
for i=1:m
    MRI_ESM1_avg4(count,1)=nanmean(MRI_ESM1_mrsosF4(i,:));
    count=count+1;
end
MRI_ESM1_Avg4=flipud(MRI_ESM1_avg4);

NorESM1_M_mrsosF1(find(MSWEP_PA_new<250))=nan;
NorESM1_M_mrsosF2(find(MSWEP_PA_new<250))=nan;
NorESM1_M_mrsosF3(find(MSWEP_PA_new<250))=nan;
NorESM1_M_mrsosF4(find(MSWEP_PA_new<250))=nan;
[m,~]=size(NorESM1_M_mrsosF1);
NorESM1_M_mrsosF1(NorESM1_M_mrsosF1==NorESM1_M_mrsosF1(m,1))=nan;
NorESM1_M_mrsosF1(1:4,:)=nan;
NorESM1_M_mrsosF1(81:m,:)=nan;
NorESM1_M_avg1(m,1)=0;
count=1;
for i=1:m
    NorESM1_M_avg1(count,1)=nanmean(NorESM1_M_mrsosF1(i,:));
    count=count+1;
end
NorESM1_M_Avg1=flipud(NorESM1_M_avg1);
NorESM1_M_mrsosF2(NorESM1_M_mrsosF2==NorESM1_M_mrsosF2(m,1))=nan;
NorESM1_M_mrsosF2(1:4,:)=nan;
NorESM1_M_mrsosF2(81:m,:)=nan;
NorESM1_M_avg2(m,1)=0;
count=1;
for i=1:m
    NorESM1_M_avg2(count,1)=nanmean(NorESM1_M_mrsosF2(i,:));
    count=count+1;
end
NorESM1_M_Avg2=flipud(NorESM1_M_avg2);
NorESM1_M_mrsosF3(NorESM1_M_mrsosF3==NorESM1_M_mrsosF3(m,1))=nan;
NorESM1_M_mrsosF3(1:4,:)=nan;
NorESM1_M_mrsosF3(81:m,:)=nan;
NorESM1_M_avg3(m,1)=0;
count=1;
for i=1:m
    NorESM1_M_avg3(count,1)=nanmean(NorESM1_M_mrsosF3(i,:));
    count=count+1;
end
NorESM1_M_Avg3=flipud(NorESM1_M_avg3);
NorESM1_M_mrsosF4(NorESM1_M_mrsosF4==NorESM1_M_mrsosF4(m,1))=nan;
NorESM1_M_mrsosF4(1:4,:)=nan;
NorESM1_M_mrsosF4(81:m,:)=nan;
NorESM1_M_avg4(m,1)=0;
count=1;
for i=1:m
    NorESM1_M_avg4(count,1)=nanmean(NorESM1_M_mrsosF4(i,:));
    count=count+1;
end
NorESM1_M_Avg4=flipud(NorESM1_M_avg4);

% BCC_CSM_AVG1=imresize(BCC_CSM_Avg1,[406 1],'bilinear');
% BCC_CSM_AVG1=imresize(BCC_CSM_avg1,[406 1],'bilinear');
% SMAP_latData=flipud(latSMAP);

% subplot(2,2,1)
% plot(SMAP_Avg1,latSMAP,'Color',[0 0 0])
% hold on
% plot(BCC_CSM_Avg1,BCC_CSM_latData,'Color',[0 0 1])
% plot(BNU_ESM_Avg1,BNU_ESM_latData,'Color',[0 1 0])
% plot(CanESM2_Avg1,CanESM2_latData,'Color',[0 1 1])
% plot(CNRM_CM5_Avg1,CNRM_CM5_latData,'Color',[1 0 0])
% plot(CSIRO_Mk_Avg1,CSIRO_Mk_latData,'Color',[1 0 1])
% plot(GFDL_CM3_Avg1,GFDL_CM3_latData,'Color',[1 1 0])
% plot(GFDL_ESM2G_Avg1,GFDL_ESM2G_latData,'Color',[1 1 1])
% plot(GFDL_ESM2M_Avg1,GFDL_ESM2M_latData,'Color',[0.1 0.1 0.1])
% plot(HadGEM2_CC_Avg1,HadGEM2_CC_latData,'Color',[0.2 0.2 0.2])
% plot(HadGEM2_ES_Avg1,HadGEM2_ES_latData,'Color',[0.3 0.3 0.3])
% plot(inmcm4_Avg1,inmcm4_latData,'Color',[0.4 0.4 0.4])
% plot(MIROC5_Avg1,MIROC5_latData,'Color',[0.5 0.5 0.5])
% plot(MIROC_ESM_Avg1,MIROC_ESM_latData,'Color',[0.6 0.6 0.6])
% plot(MIROC_ESM_CHEM_Avg1,MIROC_ESM_CHEM_latData,'Color',[0.7 0.7 0.7])
% plot(MRI_CGCM3_Avg1,MRI_CGCM3_latData,'Color',[0.8 0.8 0.8])
% plot(MRI_ESM1_Avg1,MRI_ESM1_latData,'Color',[0.9 0.9 0.9])
% plot(NorESM1_M_Avg1,NorESM1_M_latData,'Color',[0.1 0.2 0.3])
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/30<f<1/7 day^{-1}')
% % legend({'SMAP','BCC-CSM1.1','BNU-ESM','CNRM-CM5','CSIRO-Mk3.6','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','HadGEM2-CC','HadGEM2-ES','Institute for Numerical Mathematics','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MRI-CGCM3','MRI-ESM1','NorESM1-M'},'FontSize',5)
% subplot(2,2,2)
% plot(SMAP_Avg2,latSMAP,'Color',[0 0 0])
% hold on
% plot(BCC_CSM_Avg2,BCC_CSM_latData,'Color',[0 0 1])
% plot(BNU_ESM_Avg2,BNU_ESM_latData,'Color',[0 1 0])
% plot(CanESM2_Avg2,CanESM2_latData,'Color',[0 1 1])
% plot(CNRM_CM5_Avg2,CNRM_CM5_latData,'Color',[1 0 0])
% plot(CSIRO_Mk_Avg2,CSIRO_Mk_latData,'Color',[1 0 1])
% plot(GFDL_CM3_Avg2,GFDL_CM3_latData,'Color',[1 1 0])
% plot(GFDL_ESM2G_Avg2,GFDL_ESM2G_latData,'Color',[1 1 1])
% plot(GFDL_ESM2M_Avg2,GFDL_ESM2M_latData,'Color',[0.1 0.1 0.1])
% plot(HadGEM2_CC_Avg2,HadGEM2_CC_latData,'Color',[0.2 0.2 0.2])
% plot(HadGEM2_ES_Avg2,HadGEM2_ES_latData,'Color',[0.3 0.3 0.3])
% plot(inmcm4_Avg2,inmcm4_latData,'Color',[0.4 0.4 0.4])
% plot(MIROC5_Avg2,MIROC5_latData,'Color',[0.5 0.5 0.5])
% plot(MIROC_ESM_Avg2,MIROC_ESM_latData,'Color',[0.6 0.6 0.6])
% plot(MIROC_ESM_CHEM_Avg2,MIROC_ESM_CHEM_latData,'Color',[0.7 0.7 0.7])
% plot(MRI_CGCM3_Avg2,MRI_CGCM3_latData,'Color',[0.8 0.8 0.8])
% plot(MRI_ESM1_Avg2,MRI_ESM1_latData,'Color',[0.9 0.9 0.9])
% plot(NorESM1_M_Avg2,NorESM1_M_latData,'Color',[0.1 0.2 0.3])
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/90<f<1/30 day^{-1}')
% legend({'SMAP','BCC-CSM1.1','BNU-ESM','CNRM-CM5','CSIRO-Mk3.6','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','HadGEM2-CC','HadGEM2-ES','Institute for Numerical Mathematics','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MRI-CGCM3','MRI-ESM1','NorESM1-M'},'FontSize',4.7)
% subplot(2,2,3)
% plot(SMAP_Avg3,latSMAP,'Color',[0 0 0])
% hold on
% plot(BCC_CSM_Avg3,BCC_CSM_latData,'Color',[0 0 1])
% plot(BNU_ESM_Avg3,BNU_ESM_latData,'Color',[0 1 0])
% plot(CanESM2_Avg3,CanESM2_latData,'Color',[0 1 1])
% plot(CNRM_CM5_Avg3,CNRM_CM5_latData,'Color',[1 0 0])
% plot(CSIRO_Mk_Avg3,CSIRO_Mk_latData,'Color',[1 0 1])
% plot(GFDL_CM3_Avg3,GFDL_CM3_latData,'Color',[1 1 0])
% plot(GFDL_ESM2G_Avg3,GFDL_ESM2G_latData,'Color',[1 1 1])
% plot(GFDL_ESM2M_Avg3,GFDL_ESM2M_latData,'Color',[0.1 0.1 0.1])
% plot(HadGEM2_CC_Avg3,HadGEM2_CC_latData,'Color',[0.2 0.2 0.2])
% plot(HadGEM2_ES_Avg3,HadGEM2_ES_latData,'Color',[0.3 0.3 0.3])
% plot(inmcm4_Avg3,inmcm4_latData,'Color',[0.4 0.4 0.4])
% plot(MIROC5_Avg3,MIROC5_latData,'Color',[0.5 0.5 0.5])
% plot(MIROC_ESM_Avg3,MIROC_ESM_latData,'Color',[0.6 0.6 0.6])
% plot(MIROC_ESM_CHEM_Avg3,MIROC_ESM_CHEM_latData,'Color',[0.7 0.7 0.7])
% plot(MRI_CGCM3_Avg3,MRI_CGCM3_latData,'Color',[0.8 0.8 0.8])
% plot(MRI_ESM1_Avg3,MRI_ESM1_latData,'Color',[0.9 0.9 0.9])
% plot(NorESM1_M_Avg3,NorESM1_M_latData,'Color',[0.1 0.2 0.3])
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/365<f<1/90 day^{-1}')
% % legend('SMAP','BCC-CSM1.1','BNU-ESM','CNRM-CM5','CSIRO-Mk3.6','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','HadGEM2-CC','HadGEM2-ES','Institute for Numerical Mathematics','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MRI-CGCM3','MRI-ESM1','NorESM1-M')
% subplot(2,2,4)
% plot(SMAP_Avg4,latSMAP,'Color',[0 0 0])
% hold on
% plot(BCC_CSM_Avg4,BCC_CSM_latData,'Color',[0 0 1])
% plot(BNU_ESM_Avg4,BNU_ESM_latData,'Color',[0 1 0])
% plot(CanESM2_Avg4,CanESM2_latData,'Color',[0 1 1])
% plot(CNRM_CM5_Avg4,CNRM_CM5_latData,'Color',[1 0 0])
% plot(CSIRO_Mk_Avg4,CSIRO_Mk_latData,'Color',[1 0 1])
% plot(GFDL_CM3_Avg4,GFDL_CM3_latData,'Color',[1 1 0])
% plot(GFDL_ESM2G_Avg4,GFDL_ESM2G_latData,'Color',[1 1 1])
% plot(GFDL_ESM2M_Avg4,GFDL_ESM2M_latData,'Color',[0.1 0.1 0.1])
% plot(HadGEM2_CC_Avg4,HadGEM2_CC_latData,'Color',[0.2 0.2 0.2])
% plot(HadGEM2_ES_Avg4,HadGEM2_ES_latData,'Color',[0.3 0.3 0.3])
% plot(inmcm4_Avg4,inmcm4_latData,'Color',[0.4 0.4 0.4])
% plot(MIROC5_Avg4,MIROC5_latData,'Color',[0.5 0.5 0.5])
% plot(MIROC_ESM_Avg4,MIROC_ESM_latData,'Color',[0.6 0.6 0.6])
% plot(MIROC_ESM_CHEM_Avg4,MIROC_ESM_CHEM_latData,'Color',[0.7 0.7 0.7])
% plot(MRI_CGCM3_Avg4,MRI_CGCM3_latData,'Color',[0.8 0.8 0.8])
% plot(MRI_ESM1_Avg4,MRI_ESM1_latData,'Color',[0.9 0.9 0.9])
% plot(NorESM1_M_Avg4,NorESM1_M_latData,'Color',[0.1 0.2 0.3])
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('f<1/365 day^{-1}')
% % legend('SMAP','BCC-CSM1.1','BNU-ESM','CNRM-CM5','CSIRO-Mk3.6','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','HadGEM2-CC','HadGEM2-ES','Institute for Numerical Mathematics','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MRI-CGCM3','MRI-ESM1','NorESM1-M')
% sgtitle('Global soil moisture average energy spectra component with latitude')

% Change observation and models have same spatial resolution in this part
% SMAP_AVG1=[latSMAP,SMAP_Avg1];
% SMAP_AVG2=[latSMAP,SMAP_Avg2];
% SMAP_AVG3=[latSMAP,SMAP_Avg3];
% SMAP_AVG4=[latSMAP,SMAP_Avg4];

BCC_CSM_AVG1=[BCC_CSM_latData,BCC_CSM_Avg1];
[m1,~]=size(BCC_CSM_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
BCC_CSM_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(BCC_CSM_AVG1(i,1)-SMAP_AVG1(j,1))+abs(BCC_CSM_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    BCC_CSM_AVG1_new(j,1)=BCC_CSM_AVG1(idx,2);
end
BCC_CSM_AVG2=[BCC_CSM_latData,BCC_CSM_Avg2];
[m1,~]=size(BCC_CSM_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
BCC_CSM_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(BCC_CSM_AVG2(i,1)-SMAP_AVG2(j,1))+abs(BCC_CSM_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    BCC_CSM_AVG2_new(j,1)=BCC_CSM_AVG2(idx,2);
end
BCC_CSM_AVG3=[BCC_CSM_latData,BCC_CSM_Avg3];
[m1,~]=size(BCC_CSM_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
BCC_CSM_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(BCC_CSM_AVG3(i,1)-SMAP_AVG3(j,1))+abs(BCC_CSM_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    BCC_CSM_AVG3_new(j,1)=BCC_CSM_AVG3(idx,2);
end
BCC_CSM_AVG4=[BCC_CSM_latData,BCC_CSM_Avg4];
[m1,~]=size(BCC_CSM_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
BCC_CSM_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(BCC_CSM_AVG4(i,1)-SMAP_AVG4(j,1))+abs(BCC_CSM_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    BCC_CSM_AVG4_new(j,1)=BCC_CSM_AVG4(idx,2);
end

BNU_ESM_AVG1=[BNU_ESM_latData,BNU_ESM_Avg1];
[m1,~]=size(BNU_ESM_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
BNU_ESM_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(BNU_ESM_AVG1(i,1)-SMAP_AVG1(j,1))+abs(BNU_ESM_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    BNU_ESM_AVG1_new(j,1)=BNU_ESM_AVG1(idx,2);
end
BNU_ESM_AVG2=[BNU_ESM_latData,BNU_ESM_Avg2];
[m1,~]=size(BNU_ESM_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
BNU_ESM_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(BNU_ESM_AVG2(i,1)-SMAP_AVG2(j,1))+abs(BNU_ESM_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    BNU_ESM_AVG2_new(j,1)=BNU_ESM_AVG2(idx,2);
end
BNU_ESM_AVG3=[BNU_ESM_latData,BNU_ESM_Avg3];
[m1,~]=size(BNU_ESM_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
BNU_ESM_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(BNU_ESM_AVG3(i,1)-SMAP_AVG3(j,1))+abs(BNU_ESM_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    BNU_ESM_AVG3_new(j,1)=BNU_ESM_AVG3(idx,2);
end
BNU_ESM_AVG4=[BNU_ESM_latData,BNU_ESM_Avg4];
[m1,~]=size(BNU_ESM_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
BNU_ESM_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(BNU_ESM_AVG4(i,1)-SMAP_AVG4(j,1))+abs(BNU_ESM_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    BNU_ESM_AVG4_new(j,1)=BNU_ESM_AVG4(idx,2);
end

CanESM2_AVG1=[CanESM2_latData,CanESM2_Avg1];
[m1,~]=size(CanESM2_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
CanESM2_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CanESM2_AVG1(i,1)-SMAP_AVG1(j,1))+abs(CanESM2_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CanESM2_AVG1_new(j,1)=CanESM2_AVG1(idx,2);
end
CanESM2_AVG2=[CanESM2_latData,CanESM2_Avg2];
[m1,~]=size(CanESM2_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
CanESM2_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CanESM2_AVG2(i,1)-SMAP_AVG2(j,1))+abs(CanESM2_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CanESM2_AVG2_new(j,1)=CanESM2_AVG2(idx,2);
end
CanESM2_AVG3=[CanESM2_latData,CanESM2_Avg3];
[m1,~]=size(CanESM2_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
CanESM2_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CanESM2_AVG3(i,1)-SMAP_AVG3(j,1))+abs(CanESM2_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CanESM2_AVG3_new(j,1)=CanESM2_AVG3(idx,2);
end
CanESM2_AVG4=[CanESM2_latData,CanESM2_Avg4];
[m1,~]=size(CanESM2_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
CanESM2_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CanESM2_AVG4(i,1)-SMAP_AVG4(j,1))+abs(CanESM2_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CanESM2_AVG4_new(j,1)=CanESM2_AVG4(idx,2);
end

CNRM_CM5_AVG1=[CNRM_CM5_latData,CNRM_CM5_Avg1];
[m1,~]=size(CNRM_CM5_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
CNRM_CM5_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CNRM_CM5_AVG1(i,1)-SMAP_AVG1(j,1))+abs(CNRM_CM5_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CNRM_CM5_AVG1_new(j,1)=CNRM_CM5_AVG1(idx,2);
end
CNRM_CM5_AVG2=[CNRM_CM5_latData,CNRM_CM5_Avg2];
[m1,~]=size(CNRM_CM5_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
CNRM_CM5_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CNRM_CM5_AVG2(i,1)-SMAP_AVG2(j,1))+abs(CNRM_CM5_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CNRM_CM5_AVG2_new(j,1)=CNRM_CM5_AVG2(idx,2);
end
CNRM_CM5_AVG3=[CNRM_CM5_latData,CNRM_CM5_Avg3];
[m1,~]=size(CNRM_CM5_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
CNRM_CM5_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CNRM_CM5_AVG3(i,1)-SMAP_AVG3(j,1))+abs(CNRM_CM5_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CNRM_CM5_AVG3_new(j,1)=CNRM_CM5_AVG3(idx,2);
end
CNRM_CM5_AVG4=[CNRM_CM5_latData,CNRM_CM5_Avg4];
[m1,~]=size(CNRM_CM5_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
CNRM_CM5_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CNRM_CM5_AVG4(i,1)-SMAP_AVG4(j,1))+abs(CNRM_CM5_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CNRM_CM5_AVG4_new(j,1)=CNRM_CM5_AVG4(idx,2);
end

CSIRO_Mk_AVG1=[CSIRO_Mk_latData,CSIRO_Mk_Avg1];
[m1,~]=size(CSIRO_Mk_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
CSIRO_Mk_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CSIRO_Mk_AVG1(i,1)-SMAP_AVG1(j,1))+abs(CSIRO_Mk_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CSIRO_Mk_AVG1_new(j,1)=CSIRO_Mk_AVG1(idx,2);
end
CSIRO_Mk_AVG2=[CSIRO_Mk_latData,CSIRO_Mk_Avg2];
[m1,~]=size(CSIRO_Mk_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
CSIRO_Mk_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CSIRO_Mk_AVG2(i,1)-SMAP_AVG2(j,1))+abs(CSIRO_Mk_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CSIRO_Mk_AVG2_new(j,1)=CSIRO_Mk_AVG2(idx,2);
end
CSIRO_Mk_AVG3=[CSIRO_Mk_latData,CSIRO_Mk_Avg3];
[m1,~]=size(CSIRO_Mk_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
CSIRO_Mk_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CSIRO_Mk_AVG3(i,1)-SMAP_AVG3(j,1))+abs(CSIRO_Mk_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CSIRO_Mk_AVG3_new(j,1)=CSIRO_Mk_AVG3(idx,2);
end
CSIRO_Mk_AVG4=[CSIRO_Mk_latData,CSIRO_Mk_Avg4];
[m1,~]=size(CSIRO_Mk_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
CSIRO_Mk_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(CSIRO_Mk_AVG4(i,1)-SMAP_AVG4(j,1))+abs(CSIRO_Mk_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    CSIRO_Mk_AVG4_new(j,1)=CSIRO_Mk_AVG4(idx,2);
end

GFDL_CM3_AVG1=[GFDL_CM3_latData,GFDL_CM3_Avg1];
[m1,~]=size(GFDL_CM3_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
GFDL_CM3_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_CM3_AVG1(i,1)-SMAP_AVG1(j,1))+abs(GFDL_CM3_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_CM3_AVG1_new(j,1)=GFDL_CM3_AVG1(idx,2);
end
GFDL_CM3_AVG2=[GFDL_CM3_latData,GFDL_CM3_Avg2];
[m1,~]=size(GFDL_CM3_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
GFDL_CM3_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_CM3_AVG2(i,1)-SMAP_AVG2(j,1))+abs(GFDL_CM3_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_CM3_AVG2_new(j,1)=GFDL_CM3_AVG2(idx,2);
end
GFDL_CM3_AVG3=[GFDL_CM3_latData,GFDL_CM3_Avg3];
[m1,~]=size(GFDL_CM3_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
GFDL_CM3_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_CM3_AVG3(i,1)-SMAP_AVG3(j,1))+abs(GFDL_CM3_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_CM3_AVG3_new(j,1)=GFDL_CM3_AVG3(idx,2);
end
GFDL_CM3_AVG4=[GFDL_CM3_latData,GFDL_CM3_Avg4];
[m1,~]=size(GFDL_CM3_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
GFDL_CM3_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_CM3_AVG4(i,1)-SMAP_AVG4(j,1))+abs(GFDL_CM3_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_CM3_AVG4_new(j,1)=GFDL_CM3_AVG4(idx,2);
end

GFDL_ESM2G_AVG1=[GFDL_ESM2G_latData,GFDL_ESM2G_Avg1];
[m1,~]=size(GFDL_ESM2G_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
GFDL_ESM2G_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_ESM2G_AVG1(i,1)-SMAP_AVG1(j,1))+abs(GFDL_ESM2G_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_ESM2G_AVG1_new(j,1)=GFDL_ESM2G_AVG1(idx,2);
end
GFDL_ESM2G_AVG2=[GFDL_ESM2G_latData,GFDL_ESM2G_Avg2];
[m1,~]=size(GFDL_ESM2G_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
GFDL_ESM2G_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_ESM2G_AVG2(i,1)-SMAP_AVG2(j,1))+abs(GFDL_ESM2G_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_ESM2G_AVG2_new(j,1)=GFDL_ESM2G_AVG2(idx,2);
end
GFDL_ESM2G_AVG3=[GFDL_ESM2G_latData,GFDL_ESM2G_Avg3];
[m1,~]=size(GFDL_ESM2G_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
GFDL_ESM2G_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_ESM2G_AVG3(i,1)-SMAP_AVG3(j,1))+abs(GFDL_ESM2G_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_ESM2G_AVG3_new(j,1)=GFDL_ESM2G_AVG3(idx,2);
end
GFDL_ESM2G_AVG4=[GFDL_ESM2G_latData,GFDL_ESM2G_Avg4];
[m1,~]=size(GFDL_ESM2G_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
GFDL_ESM2G_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_ESM2G_AVG4(i,1)-SMAP_AVG4(j,1))+abs(GFDL_ESM2G_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_ESM2G_AVG4_new(j,1)=GFDL_ESM2G_AVG4(idx,2);
end

GFDL_ESM2M_AVG1=[GFDL_ESM2M_latData,GFDL_ESM2M_Avg1];
[m1,~]=size(GFDL_ESM2M_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
GFDL_ESM2M_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_ESM2M_AVG1(i,1)-SMAP_AVG1(j,1))+abs(GFDL_ESM2M_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_ESM2M_AVG1_new(j,1)=GFDL_ESM2M_AVG1(idx,2);
end
GFDL_ESM2M_AVG2=[GFDL_ESM2M_latData,GFDL_ESM2M_Avg2];
[m1,~]=size(GFDL_ESM2M_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
GFDL_ESM2M_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_ESM2M_AVG2(i,1)-SMAP_AVG2(j,1))+abs(GFDL_ESM2M_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_ESM2M_AVG2_new(j,1)=GFDL_ESM2M_AVG2(idx,2);
end
GFDL_ESM2M_AVG3=[GFDL_ESM2M_latData,GFDL_ESM2M_Avg3];
[m1,~]=size(GFDL_ESM2M_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
GFDL_ESM2M_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_ESM2M_AVG3(i,1)-SMAP_AVG3(j,1))+abs(GFDL_ESM2M_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_ESM2M_AVG3_new(j,1)=GFDL_ESM2M_AVG3(idx,2);
end
GFDL_ESM2M_AVG4=[GFDL_ESM2M_latData,GFDL_ESM2M_Avg4];
[m1,~]=size(GFDL_ESM2M_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
GFDL_ESM2M_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(GFDL_ESM2M_AVG4(i,1)-SMAP_AVG4(j,1))+abs(GFDL_ESM2M_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    GFDL_ESM2M_AVG4_new(j,1)=GFDL_ESM2M_AVG4(idx,2);
end

HadGEM2_CC_AVG1=[HadGEM2_CC_latData,HadGEM2_CC_Avg1];
[m1,~]=size(HadGEM2_CC_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
HadGEM2_CC_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(HadGEM2_CC_AVG1(i,1)-SMAP_AVG1(j,1))+abs(HadGEM2_CC_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    HadGEM2_CC_AVG1_new(j,1)=HadGEM2_CC_AVG1(idx,2);
end
HadGEM2_CC_AVG2=[HadGEM2_CC_latData,HadGEM2_CC_Avg2];
[m1,~]=size(HadGEM2_CC_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
HadGEM2_CC_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(HadGEM2_CC_AVG2(i,1)-SMAP_AVG2(j,1))+abs(HadGEM2_CC_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    HadGEM2_CC_AVG2_new(j,1)=HadGEM2_CC_AVG2(idx,2);
end
HadGEM2_CC_AVG3=[HadGEM2_CC_latData,HadGEM2_CC_Avg3];
[m1,~]=size(HadGEM2_CC_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
HadGEM2_CC_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(HadGEM2_CC_AVG3(i,1)-SMAP_AVG3(j,1))+abs(HadGEM2_CC_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    HadGEM2_CC_AVG3_new(j,1)=HadGEM2_CC_AVG3(idx,2);
end
HadGEM2_CC_AVG4=[HadGEM2_CC_latData,HadGEM2_CC_Avg4];
[m1,~]=size(HadGEM2_CC_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
HadGEM2_CC_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(HadGEM2_CC_AVG4(i,1)-SMAP_AVG4(j,1))+abs(HadGEM2_CC_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    HadGEM2_CC_AVG4_new(j,1)=HadGEM2_CC_AVG4(idx,2);
end

HadGEM2_ES_AVG1=[HadGEM2_ES_latData,HadGEM2_ES_Avg1];
[m1,~]=size(HadGEM2_ES_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
HadGEM2_ES_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(HadGEM2_ES_AVG1(i,1)-SMAP_AVG1(j,1))+abs(HadGEM2_ES_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    HadGEM2_ES_AVG1_new(j,1)=HadGEM2_ES_AVG1(idx,2);
end
HadGEM2_ES_AVG2=[HadGEM2_ES_latData,HadGEM2_ES_Avg2];
[m1,~]=size(HadGEM2_ES_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
HadGEM2_ES_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(HadGEM2_ES_AVG2(i,1)-SMAP_AVG2(j,1))+abs(HadGEM2_ES_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    HadGEM2_ES_AVG2_new(j,1)=HadGEM2_ES_AVG2(idx,2);
end
HadGEM2_ES_AVG3=[HadGEM2_ES_latData,HadGEM2_ES_Avg3];
[m1,~]=size(HadGEM2_ES_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
HadGEM2_ES_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(HadGEM2_ES_AVG3(i,1)-SMAP_AVG3(j,1))+abs(HadGEM2_ES_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    HadGEM2_ES_AVG3_new(j,1)=HadGEM2_ES_AVG3(idx,2);
end
HadGEM2_ES_AVG4=[HadGEM2_ES_latData,HadGEM2_ES_Avg4];
[m1,~]=size(HadGEM2_ES_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
HadGEM2_ES_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(HadGEM2_ES_AVG4(i,1)-SMAP_AVG4(j,1))+abs(HadGEM2_ES_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    HadGEM2_ES_AVG4_new(j,1)=HadGEM2_ES_AVG4(idx,2);
end

inmcm4_AVG1=[inmcm4_latData,inmcm4_Avg1];
[m1,~]=size(inmcm4_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
inmcm4_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(inmcm4_AVG1(i,1)-SMAP_AVG1(j,1))+abs(inmcm4_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    inmcm4_AVG1_new(j,1)=inmcm4_AVG1(idx,2);
end
inmcm4_AVG2=[inmcm4_latData,inmcm4_Avg2];
[m1,~]=size(inmcm4_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
inmcm4_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(inmcm4_AVG2(i,1)-SMAP_AVG2(j,1))+abs(inmcm4_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    inmcm4_AVG2_new(j,1)=inmcm4_AVG2(idx,2);
end
inmcm4_AVG3=[inmcm4_latData,inmcm4_Avg3];
[m1,~]=size(inmcm4_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
inmcm4_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(inmcm4_AVG3(i,1)-SMAP_AVG3(j,1))+abs(inmcm4_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    inmcm4_AVG3_new(j,1)=inmcm4_AVG3(idx,2);
end
inmcm4_AVG4=[inmcm4_latData,inmcm4_Avg4];
[m1,~]=size(inmcm4_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
inmcm4_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(inmcm4_AVG4(i,1)-SMAP_AVG4(j,1))+abs(inmcm4_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    inmcm4_AVG4_new(j,1)=inmcm4_AVG4(idx,2);
end

MIROC5_AVG1=[MIROC5_latData,MIROC5_Avg1];
[m1,~]=size(MIROC5_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
MIROC5_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC5_AVG1(i,1)-SMAP_AVG1(j,1))+abs(MIROC5_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC5_AVG1_new(j,1)=MIROC5_AVG1(idx,2);
end
MIROC5_AVG2=[MIROC5_latData,MIROC5_Avg2];
[m1,~]=size(MIROC5_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
MIROC5_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC5_AVG2(i,1)-SMAP_AVG2(j,1))+abs(MIROC5_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC5_AVG2_new(j,1)=MIROC5_AVG2(idx,2);
end
MIROC5_AVG3=[MIROC5_latData,MIROC5_Avg3];
[m1,~]=size(MIROC5_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
MIROC5_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC5_AVG3(i,1)-SMAP_AVG3(j,1))+abs(MIROC5_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC5_AVG3_new(j,1)=MIROC5_AVG3(idx,2);
end
MIROC5_AVG4=[MIROC5_latData,MIROC5_Avg4];
[m1,~]=size(MIROC5_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
MIROC5_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC5_AVG4(i,1)-SMAP_AVG4(j,1))+abs(MIROC5_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC5_AVG4_new(j,1)=MIROC5_AVG4(idx,2);
end

MIROC_ESM_AVG1=[MIROC_ESM_latData,MIROC_ESM_Avg1];
[m1,~]=size(MIROC_ESM_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
MIROC_ESM_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC_ESM_AVG1(i,1)-SMAP_AVG1(j,1))+abs(MIROC_ESM_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC_ESM_AVG1_new(j,1)=MIROC_ESM_AVG1(idx,2);
end
MIROC_ESM_AVG2=[MIROC_ESM_latData,MIROC_ESM_Avg2];
[m1,~]=size(MIROC_ESM_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
MIROC_ESM_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC_ESM_AVG2(i,1)-SMAP_AVG2(j,1))+abs(MIROC_ESM_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC_ESM_AVG2_new(j,1)=MIROC_ESM_AVG2(idx,2);
end
MIROC_ESM_AVG3=[MIROC_ESM_latData,MIROC_ESM_Avg3];
[m1,~]=size(MIROC_ESM_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
MIROC_ESM_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC_ESM_AVG3(i,1)-SMAP_AVG3(j,1))+abs(MIROC_ESM_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC_ESM_AVG3_new(j,1)=MIROC_ESM_AVG3(idx,2);
end
MIROC_ESM_AVG4=[MIROC_ESM_latData,MIROC_ESM_Avg4];
[m1,~]=size(MIROC_ESM_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
MIROC_ESM_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC_ESM_AVG4(i,1)-SMAP_AVG4(j,1))+abs(MIROC_ESM_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC_ESM_AVG4_new(j,1)=MIROC_ESM_AVG4(idx,2);
end

MIROC_ESM_CHEM_AVG1=[MIROC_ESM_CHEM_latData,MIROC_ESM_CHEM_Avg1];
[m1,~]=size(MIROC_ESM_CHEM_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
MIROC_ESM_CHEM_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC_ESM_CHEM_AVG1(i,1)-SMAP_AVG1(j,1))+abs(MIROC_ESM_CHEM_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC_ESM_CHEM_AVG1_new(j,1)=MIROC_ESM_CHEM_AVG1(idx,2);
end
MIROC_ESM_CHEM_AVG2=[MIROC_ESM_CHEM_latData,MIROC_ESM_CHEM_Avg2];
[m1,~]=size(MIROC_ESM_CHEM_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
MIROC_ESM_CHEM_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC_ESM_CHEM_AVG2(i,1)-SMAP_AVG2(j,1))+abs(MIROC_ESM_CHEM_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC_ESM_CHEM_AVG2_new(j,1)=MIROC_ESM_CHEM_AVG2(idx,2);
end
MIROC_ESM_CHEM_AVG3=[MIROC_ESM_CHEM_latData,MIROC_ESM_CHEM_Avg3];
[m1,~]=size(MIROC_ESM_CHEM_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
MIROC_ESM_CHEM_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC_ESM_CHEM_AVG3(i,1)-SMAP_AVG3(j,1))+abs(MIROC_ESM_CHEM_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC_ESM_CHEM_AVG3_new(j,1)=MIROC_ESM_CHEM_AVG3(idx,2);
end
MIROC_ESM_CHEM_AVG4=[MIROC_ESM_CHEM_latData,MIROC_ESM_CHEM_Avg4];
[m1,~]=size(MIROC_ESM_CHEM_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
MIROC_ESM_CHEM_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MIROC_ESM_CHEM_AVG4(i,1)-SMAP_AVG4(j,1))+abs(MIROC_ESM_CHEM_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MIROC_ESM_CHEM_AVG4_new(j,1)=MIROC_ESM_CHEM_AVG4(idx,2);
end

MRI_CGCM3_AVG1=[MRI_CGCM3_latData,MRI_CGCM3_Avg1];
[m1,~]=size(MRI_CGCM3_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
MRI_CGCM3_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MRI_CGCM3_AVG1(i,1)-SMAP_AVG1(j,1))+abs(MRI_CGCM3_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MRI_CGCM3_AVG1_new(j,1)=MRI_CGCM3_AVG1(idx,2);
end
MRI_CGCM3_AVG2=[MRI_CGCM3_latData,MRI_CGCM3_Avg2];
[m1,~]=size(MRI_CGCM3_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
MRI_CGCM3_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MRI_CGCM3_AVG2(i,1)-SMAP_AVG2(j,1))+abs(MRI_CGCM3_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MRI_CGCM3_AVG2_new(j,1)=MRI_CGCM3_AVG2(idx,2);
end
MRI_CGCM3_AVG3=[MRI_CGCM3_latData,MRI_CGCM3_Avg3];
[m1,~]=size(MRI_CGCM3_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
MRI_CGCM3_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MRI_CGCM3_AVG3(i,1)-SMAP_AVG3(j,1))+abs(MRI_CGCM3_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MRI_CGCM3_AVG3_new(j,1)=MRI_CGCM3_AVG3(idx,2);
end
MRI_CGCM3_AVG4=[MRI_CGCM3_latData,MRI_CGCM3_Avg4];
[m1,~]=size(MRI_CGCM3_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
MRI_CGCM3_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MRI_CGCM3_AVG4(i,1)-SMAP_AVG4(j,1))+abs(MRI_CGCM3_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MRI_CGCM3_AVG4_new(j,1)=MRI_CGCM3_AVG4(idx,2);
end

MRI_ESM1_AVG1=[MRI_ESM1_latData,MRI_ESM1_Avg1];
[m1,~]=size(MRI_ESM1_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
MRI_ESM1_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MRI_ESM1_AVG1(i,1)-SMAP_AVG1(j,1))+abs(MRI_ESM1_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MRI_ESM1_AVG1_new(j,1)=MRI_ESM1_AVG1(idx,2);
end
MRI_ESM1_AVG2=[MRI_ESM1_latData,MRI_ESM1_Avg2];
[m1,~]=size(MRI_ESM1_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
MRI_ESM1_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MRI_ESM1_AVG2(i,1)-SMAP_AVG2(j,1))+abs(MRI_ESM1_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MRI_ESM1_AVG2_new(j,1)=MRI_ESM1_AVG2(idx,2);
end
MRI_ESM1_AVG3=[MRI_ESM1_latData,MRI_ESM1_Avg3];
[m1,~]=size(MRI_ESM1_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
MRI_ESM1_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MRI_ESM1_AVG3(i,1)-SMAP_AVG3(j,1))+abs(MRI_ESM1_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MRI_ESM1_AVG3_new(j,1)=MRI_ESM1_AVG3(idx,2);
end
MRI_ESM1_AVG4=[MRI_ESM1_latData,MRI_ESM1_Avg4];
[m1,~]=size(MRI_ESM1_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
MRI_ESM1_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(MRI_ESM1_AVG4(i,1)-SMAP_AVG4(j,1))+abs(MRI_ESM1_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    MRI_ESM1_AVG4_new(j,1)=MRI_ESM1_AVG4(idx,2);
end

NorESM1_M_AVG1=[NorESM1_M_latData,NorESM1_M_Avg1];
[m1,~]=size(NorESM1_M_AVG1);
[m2,~]=size(SMAP_AVG1);
dif(m1,1)=0;
NorESM1_M_AVG1_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(NorESM1_M_AVG1(i,1)-SMAP_AVG1(j,1))+abs(NorESM1_M_AVG1(i,2)-SMAP_AVG1(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    NorESM1_M_AVG1_new(j,1)=NorESM1_M_AVG1(idx,2);
end
NorESM1_M_AVG2=[NorESM1_M_latData,NorESM1_M_Avg2];
[m1,~]=size(NorESM1_M_AVG2);
[m2,~]=size(SMAP_AVG2);
dif(m1,1)=0;
NorESM1_M_AVG2_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(NorESM1_M_AVG2(i,1)-SMAP_AVG2(j,1))+abs(NorESM1_M_AVG2(i,2)-SMAP_AVG2(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    NorESM1_M_AVG2_new(j,1)=NorESM1_M_AVG2(idx,2);
end
NorESM1_M_AVG3=[NorESM1_M_latData,NorESM1_M_Avg3];
[m1,~]=size(NorESM1_M_AVG3);
[m2,~]=size(SMAP_AVG3);
dif(m1,1)=0;
NorESM1_M_AVG3_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(NorESM1_M_AVG3(i,1)-SMAP_AVG3(j,1))+abs(NorESM1_M_AVG3(i,2)-SMAP_AVG3(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    NorESM1_M_AVG3_new(j,1)=NorESM1_M_AVG3(idx,2);
end
NorESM1_M_AVG4=[NorESM1_M_latData,NorESM1_M_Avg4];
[m1,~]=size(NorESM1_M_AVG4);
[m2,~]=size(SMAP_AVG4);
dif(m1,1)=0;
NorESM1_M_AVG4_new(m2,1)=0;
for j=1:m2
    count=1;
    for i=1:m1
        dif(count,1)=abs(NorESM1_M_AVG4(i,1)-SMAP_AVG4(j,1))+abs(NorESM1_M_AVG4(i,2)-SMAP_AVG4(j,2));
        count=count+1;
    end
    [~,idx]=min(dif);
    NorESM1_M_AVG4_new(j,1)=NorESM1_M_AVG4(idx,2);
end

AVG1_new=(BCC_CSM_AVG1_new+BNU_ESM_AVG1_new+CanESM2_AVG1_new+CNRM_CM5_AVG1_new+CSIRO_Mk_AVG1_new+GFDL_CM3_AVG1_new+GFDL_ESM2G_AVG1_new+GFDL_ESM2M_AVG1_new+HadGEM2_CC_AVG1_new+HadGEM2_ES_AVG1_new+inmcm4_AVG1_new+MIROC5_AVG1_new+MIROC_ESM_AVG1_new+MIROC_ESM_CHEM_AVG1_new+MRI_CGCM3_AVG1_new+MRI_ESM1_AVG1_new+NorESM1_M_AVG1_new)/17;
AVG2_new=(BCC_CSM_AVG2_new+BNU_ESM_AVG2_new+CanESM2_AVG2_new+CNRM_CM5_AVG2_new+CSIRO_Mk_AVG2_new+GFDL_CM3_AVG2_new+GFDL_ESM2G_AVG2_new+GFDL_ESM2M_AVG2_new+HadGEM2_CC_AVG2_new+HadGEM2_ES_AVG2_new+inmcm4_AVG2_new+MIROC5_AVG2_new+MIROC_ESM_AVG2_new+MIROC_ESM_CHEM_AVG2_new+MRI_CGCM3_AVG2_new+MRI_ESM1_AVG2_new+NorESM1_M_AVG2_new)/17;
AVG3_new=(BCC_CSM_AVG3_new+BNU_ESM_AVG3_new+CanESM2_AVG3_new+CNRM_CM5_AVG3_new+CSIRO_Mk_AVG3_new+GFDL_CM3_AVG3_new+GFDL_ESM2G_AVG3_new+GFDL_ESM2M_AVG3_new+HadGEM2_CC_AVG3_new+HadGEM2_ES_AVG3_new+inmcm4_AVG3_new+MIROC5_AVG3_new+MIROC_ESM_AVG3_new+MIROC_ESM_CHEM_AVG3_new+MRI_CGCM3_AVG3_new+MRI_ESM1_AVG3_new+NorESM1_M_AVG3_new)/17;
AVG4_new=(BCC_CSM_AVG4_new+BNU_ESM_AVG4_new+CanESM2_AVG4_new+CNRM_CM5_AVG4_new+CSIRO_Mk_AVG4_new+GFDL_CM3_AVG4_new+GFDL_ESM2G_AVG4_new+GFDL_ESM2M_AVG4_new+HadGEM2_CC_AVG4_new+HadGEM2_ES_AVG4_new+inmcm4_AVG4_new+MIROC5_AVG4_new+MIROC_ESM_AVG4_new+MIROC_ESM_CHEM_AVG4_new+MRI_CGCM3_AVG4_new+MRI_ESM1_AVG4_new+NorESM1_M_AVG4_new)/17;
count=1;
std_AVG1_new(406,1)=0;
for i=1:406
    std_f1(1,:)=BCC_CSM_AVG1_new(i,:);
    std_f1(2,:)=BNU_ESM_AVG1_new(i,:);
    std_f1(3,:)=CanESM2_AVG1_new(i,:);
    std_f1(4,:)=CNRM_CM5_AVG1_new(i,:);
    std_f1(5,:)=CSIRO_Mk_AVG1_new(i,:);
    std_f1(6,:)=GFDL_CM3_AVG1_new(i,:);
    std_f1(7,:)=GFDL_ESM2G_AVG1_new(i,:);
    std_f1(8,:)=GFDL_ESM2M_AVG1_new(i,:);
    std_f1(9,:)=HadGEM2_CC_AVG1_new(i,:);
    std_f1(10,:)=HadGEM2_ES_AVG1_new(i,:);
    std_f1(11,:)=inmcm4_AVG1_new(i,:);
    std_f1(12,:)=MIROC5_AVG1_new(i,:);
    std_f1(13,:)=MIROC_ESM_AVG1_new(i,:);
    std_f1(14,:)=MIROC_ESM_CHEM_AVG1_new(i,:);
    std_f1(15,:)=MRI_CGCM3_AVG1_new(i,:);
    std_f1(16,:)=MRI_ESM1_AVG1_new(i,:);
    std_f1(17,:)=NorESM1_M_AVG1_new(i,:);
    std_AVG1=std(std_f1);
    std_AVG1_new(count,:)=std_AVG1;
    count=count+1;
end
count=1;
std_AVG2_new(406,1)=0;
for i=1:406
    std_f2(1,:)=BCC_CSM_AVG2_new(i,:);
    std_f2(2,:)=BNU_ESM_AVG2_new(i,:);
    std_f2(3,:)=CanESM2_AVG2_new(i,:);
    std_f2(4,:)=CNRM_CM5_AVG2_new(i,:);
    std_f2(5,:)=CSIRO_Mk_AVG2_new(i,:);
    std_f2(6,:)=GFDL_CM3_AVG2_new(i,:);
    std_f2(7,:)=GFDL_ESM2G_AVG2_new(i,:);
    std_f2(8,:)=GFDL_ESM2M_AVG2_new(i,:);
    std_f2(9,:)=HadGEM2_CC_AVG2_new(i,:);
    std_f2(10,:)=HadGEM2_ES_AVG2_new(i,:);
    std_f2(11,:)=inmcm4_AVG2_new(i,:);
    std_f2(12,:)=MIROC5_AVG2_new(i,:);
    std_f2(13,:)=MIROC_ESM_AVG2_new(i,:);
    std_f2(14,:)=MIROC_ESM_CHEM_AVG2_new(i,:);
    std_f2(15,:)=MRI_CGCM3_AVG2_new(i,:);
    std_f2(16,:)=MRI_ESM1_AVG2_new(i,:);
    std_f2(17,:)=NorESM1_M_AVG2_new(i,:);
    std_AVG2=std(std_f2);
    std_AVG2_new(count,:)=std_AVG2;
    count=count+1;
end
count=1;
std_AVG3_new(406,1)=0;
for i=1:406
    std_f3(1,:)=BCC_CSM_AVG3_new(i,:);
    std_f3(2,:)=BNU_ESM_AVG3_new(i,:);
    std_f3(3,:)=CanESM2_AVG3_new(i,:);
    std_f3(4,:)=CNRM_CM5_AVG3_new(i,:);
    std_f3(5,:)=CSIRO_Mk_AVG3_new(i,:);
    std_f3(6,:)=GFDL_CM3_AVG3_new(i,:);
    std_f3(7,:)=GFDL_ESM2G_AVG3_new(i,:);
    std_f3(8,:)=GFDL_ESM2M_AVG3_new(i,:);
    std_f3(9,:)=HadGEM2_CC_AVG3_new(i,:);
    std_f3(10,:)=HadGEM2_ES_AVG3_new(i,:);
    std_f3(11,:)=inmcm4_AVG3_new(i,:);
    std_f3(12,:)=MIROC5_AVG3_new(i,:);
    std_f3(13,:)=MIROC_ESM_AVG3_new(i,:);
    std_f3(14,:)=MIROC_ESM_CHEM_AVG3_new(i,:);
    std_f3(15,:)=MRI_CGCM3_AVG3_new(i,:);
    std_f3(16,:)=MRI_ESM1_AVG3_new(i,:);
    std_f3(17,:)=NorESM1_M_AVG3_new(i,:);
    std_AVG3=std(std_f3);
    std_AVG3_new(count,:)=std_AVG3;
    count=count+1;
end
count=1;
std_AVG4_new(406,1)=0;
for i=1:406
    std_f4(1,:)=BCC_CSM_AVG4_new(i,:);
    std_f4(2,:)=BNU_ESM_AVG4_new(i,:);
    std_f4(3,:)=CanESM2_AVG4_new(i,:);
    std_f4(4,:)=CNRM_CM5_AVG4_new(i,:);
    std_f4(5,:)=CSIRO_Mk_AVG4_new(i,:);
    std_f4(6,:)=GFDL_CM3_AVG4_new(i,:);
    std_f4(7,:)=GFDL_ESM2G_AVG4_new(i,:);
    std_f4(8,:)=GFDL_ESM2M_AVG4_new(i,:);
    std_f4(9,:)=HadGEM2_CC_AVG4_new(i,:);
    std_f4(10,:)=HadGEM2_ES_AVG4_new(i,:);
    std_f4(11,:)=inmcm4_AVG4_new(i,:);
    std_f4(12,:)=MIROC5_AVG4_new(i,:);
    std_f4(13,:)=MIROC_ESM_AVG4_new(i,:);
    std_f4(14,:)=MIROC_ESM_CHEM_AVG4_new(i,:);
    std_f4(15,:)=MRI_CGCM3_AVG4_new(i,:);
    std_f4(16,:)=MRI_ESM1_AVG4_new(i,:);
    std_f4(17,:)=NorESM1_M_AVG4_new(i,:);
    std_AVG4=std(std_f4);
    std_AVG4_new(count,:)=std_AVG4;
    count=count+1;
end

% % Final result map method 1
% subplot(2,2,1)
% plot(SMAP_Avg1,latSMAP,'Color',[0 0 0],'LineWidth',1)
% hold on
% plot(AVG1_new,latSMAP,'Color',[0 0 1],'LineWidth',1)
% x11=AVG1_new-std_AVG1_new;
% x12=AVG1_new+std_AVG1_new;
% d=1;
% y1=latSMAP-d/2;
% y2=latSMAP+d/2;
% nn1=nan(length(AVG1_new),1);
% ex1=[x11(:) x12(:) nn1 x11(:) x11(:) nn1 x12(:) x12(:) nn1]';
% ey1=[latSMAP(:) latSMAP(:) nn1 y1(:) y2(:) nn1 y1(:) y2(:) nn1]';
% ex1=ex1(:);
% ey1=ey1(:); 
% plot(ex1,ey1)
% % plot(AVG1_new,latSMAP,'Color',[0 0 1])
% % h1=fill([AVG1_new-std_AVG1_new,AVG1_new+std_AVG1_new],latSMAP,'r');
% % set(h1,'FaceColor','m','FaceAlpha',0.6,'EdgeColor','r')
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/30<f<1/7 day^{-1}')
% subplot(2,2,2)
% plot(SMAP_Avg2,latSMAP,'Color',[0 0 0],'LineWidth',1)
% hold on
% plot(AVG2_new,latSMAP,'Color',[0 0 1],'LineWidth',1)
% x21=AVG2_new-std_AVG2_new;
% x22=AVG2_new+std_AVG2_new;
% d=1;
% y1=latSMAP-d/2;
% y2=latSMAP+d/2;
% nn2=nan(length(AVG1_new),1);
% ex2=[x21(:) x22(:) nn2 x21(:) x21(:) nn2 x22(:) x22(:) nn2]';
% ey2=[latSMAP(:) latSMAP(:) nn2 y1(:) y2(:) nn2 y1(:) y2(:) nn2]';
% ex2=ex2(:);
% ey2=ey2(:); 
% plot(ex2,ey2)
% % plot(AVG2_new,latSMAP,'Color',[0 0 1])
% % h2=fill([AVG2_new-std_AVG2_new,AVG2_new+std_AVG2_new],latSMAP,'r');
% % set(h2,'FaceColor','m','FaceAlpha',0.6,'EdgeColor','r')
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/90<f<1/30 day^{-1}')
% subplot(2,2,3)
% plot(SMAP_Avg3,latSMAP,'Color',[0 0 0],'LineWidth',1)
% hold on
% plot(AVG3_new,latSMAP,'Color',[0 0 1],'LineWidth',1)
% x31=AVG3_new-std_AVG3_new;
% x32=AVG3_new+std_AVG3_new;
% d=1;
% y1=latSMAP-d/2;
% y2=latSMAP+d/2;
% nn3=nan(length(AVG1_new),1);
% ex3=[x31(:) x32(:) nn3 x31(:) x31(:) nn3 x32(:) x32(:) nn3]';
% ey3=[latSMAP(:) latSMAP(:) nn3 y1(:) y2(:) nn3 y1(:) y2(:) nn3]';
% ex3=ex3(:);
% ey3=ey3(:); 
% plot(ex3,ey3)
% % plot(AVG3_new,latSMAP,'Color',[0 0 1])
% % h3=fill([AVG3_new-std_AVG3_new,AVG3_new+std_AVG3_new],latSMAP,'r');
% % set(h3,'FaceColor','m','FaceAlpha',0.6,'EdgeColor','r')
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/365<f<1/90 day^{-1}')
% subplot(2,2,4)
% plot(SMAP_Avg4,latSMAP,'Color',[0 0 0],'LineWidth',1)
% hold on
% plot(AVG4_new,latSMAP,'Color',[0 0 1],'LineWidth',1)
% x41=AVG4_new-std_AVG4_new;
% x42=AVG4_new+std_AVG4_new;
% d=1;
% y1=latSMAP-d/2;
% y2=latSMAP+d/2;
% nn4=nan(length(AVG1_new),1);
% ex4=[x41(:) x42(:) nn4 x41(:) x41(:) nn4 x42(:) x42(:) nn4]';
% ey4=[latSMAP(:) latSMAP(:) nn4 y1(:) y2(:) nn4 y1(:) y2(:) nn4]';
% ex4=ex4(:);
% ey4=ey4(:); 
% plot(ex4,ey4)
% % plot(AVG4_new,latSMAP,'Color',[0 0 1])
% % h4=fill([AVG4_new-std_AVG4_new,AVG4_new+std_AVG4_new],latSMAP,'r');
% % set(h4,'FaceColor','m','FaceAlpha',0.6,'EdgeColor','r')
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('f<1/365 day^{-1}')
% suptitle('Global soil moisture average energy spectra component with latitude')

% % Final result map method 2
subplot(2,2,1)
plot(latSMAP,SMAP_Avg1,'Color',[0 0 0],'LineWidth',0.7)
hold on
shadedErrorBar(latSMAP,AVG1_new,std_AVG1_new,'lineprops',{'-r','MarkerFaceColor','r'});
hold off
grid on
xlim([-90,90])
ylim([0,1])
xlabel('Latitude')
ylabel('Soil moisture component')
set(gca,'XTick',[-90,-60,-30,0,30,60,90])
title('1/30<f<1/7 day^{-1}')
legend('SMAP data (observation)','CMIP5 models')
subplot(2,2,2);
plot(latSMAP,SMAP_Avg2,'Color',[0 0 0],'LineWidth',0.7)
hold on
shadedErrorBar(latSMAP,AVG2_new,std_AVG2_new,'lineprops',{'-r','MarkerFaceColor','r'});
hold off
grid on
xlim([-90,90])
ylim([0,1])
xlabel('Latitude')
ylabel('Soil moisture component')
set(gca,'XTick',[-90,-60,-30,0,30,60,90])
title('1/90<f<1/30 day^{-1}')
legend('SMAP data (observation)','CMIP5 models')
subplot(2,2,3);
plot(latSMAP,SMAP_Avg3,'Color',[0 0 0],'LineWidth',0.7)
hold on
shadedErrorBar(latSMAP,AVG3_new,std_AVG3_new,'lineprops',{'-r','MarkerFaceColor','r'});
hold off
grid on
xlim([-90,90])
ylim([0,1])
xlabel('Latitude')
ylabel('Soil moisture component')
set(gca,'XTick',[-90,-60,-30,0,30,60,90])
title('1/365<f<1/90 day^{-1}')
legend('SMAP data (observation)','CMIP5 models')
subplot(2,2,4);
plot(latSMAP,SMAP_Avg4,'Color',[0 0 0],'LineWidth',0.7)
hold on
shadedErrorBar(latSMAP,AVG4_new,std_AVG4_new,'lineprops',{'-r','MarkerFaceColor','r'});
hold off
grid on
xlim([-90,90])
ylim([0,1])
xlabel('Latitude')
ylabel('Soil moisture component')
set(gca,'XTick',[-90,-60,-30,0,30,60,90])
title('f<1/365 day^{-1}')
legend('SMAP data (observation)','CMIP5 models')
sgtitle('Global soil moisture average energy spectra component with latitude')

% Final result map method 2 comparision
% subplot(2,2,1)
% plot(latSMAP,SMAP_Avg1,'Color',[0 0 0],'Linewidth',1)
% hold on
% plot(BCC_CSM_latData,BCC_CSM_Avg1,'Color',[0 0 1])
% plot(BNU_ESM_latData,BNU_ESM_Avg1,'Color',[0 1 0])
% plot(CanESM2_latData,CanESM2_Avg1,'Color',[0 1 1])
% plot(CNRM_CM5_latData,CNRM_CM5_Avg1,'Color',[1 0 0])
% plot(CSIRO_Mk_latData,CSIRO_Mk_Avg1,'Color',[1 0 1])
% plot(GFDL_CM3_latData,GFDL_CM3_Avg1,'Color',[1 1 0])
% plot(GFDL_ESM2G_latData,GFDL_ESM2G_Avg1,'Color',[1 1 1])
% plot(GFDL_ESM2M_latData,GFDL_ESM2M_Avg1,'Color',[0.1 0.1 0.1])
% plot(HadGEM2_CC_latData,HadGEM2_CC_Avg1,'Color',[0.2 0.2 0.2])
% plot(HadGEM2_ES_latData,HadGEM2_ES_Avg1,'Color',[0.3 0.3 0.3])
% plot(inmcm4_latData,inmcm4_Avg1,'Color',[0.4 0.4 0.4])
% plot(MIROC5_latData,MIROC5_Avg1,'Color',[0.5 0.5 0.5])
% plot(MIROC_ESM_latData,MIROC_ESM_Avg1,'Color',[0.6 0.6 0.6])
% plot(MIROC_ESM_CHEM_latData,MIROC_ESM_CHEM_Avg1,'Color',[0.7 0.7 0.7])
% plot(MRI_CGCM3_latData,MRI_CGCM3_Avg1,'Color',[0.8 0.8 0.8])
% plot(MRI_ESM1_latData,MRI_ESM1_Avg1,'Color',[0.9 0.9 0.9])
% plot(NorESM1_M_latData,NorESM1_M_Avg1,'Color',[0.1 0.2 0.3])
% hold off
% xlim([-90,90])
% ylim([0,1])
% xlabel('Latitude')
% ylabel('Soil moisture component')
% set(gca,'XTick',[-90,-60,-30,0,30,60,90])
% title('1/30<f<1/7 day^{-1}')
% % legend({'SMAP','BCC-CSM1.1','BNU-ESM','CNRM-CM5','CSIRO-Mk3.6','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','HadGEM2-CC','HadGEM2-ES','Institute for Numerical Mathematics','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MRI-CGCM3','MRI-ESM1','NorESM1-M'},'FontSize',5)
% subplot(2,2,2)
% plot(latSMAP,SMAP_Avg2,'Color',[0 0 0],'Linewidth',1)
% hold on
% plot(BCC_CSM_latData,BCC_CSM_Avg2,'Color',[0 0 1])
% plot(BNU_ESM_latData,BNU_ESM_Avg2,'Color',[0 1 0])
% plot(CanESM2_latData,CanESM2_Avg2,'Color',[0 1 1])
% plot(CNRM_CM5_latData,CNRM_CM5_Avg2,'Color',[1 0 0])
% plot(CSIRO_Mk_latData,CSIRO_Mk_Avg2,'Color',[1 0 1])
% plot(GFDL_CM3_latData,GFDL_CM3_Avg2,'Color',[1 1 0])
% plot(GFDL_ESM2G_latData,GFDL_ESM2G_Avg2,'Color',[1 1 1])
% plot(GFDL_ESM2M_latData,GFDL_ESM2M_Avg2,'Color',[0.1 0.1 0.1])
% plot(HadGEM2_CC_latData,HadGEM2_CC_Avg2,'Color',[0.2 0.2 0.2])
% plot(HadGEM2_ES_latData,HadGEM2_ES_Avg2,'Color',[0.3 0.3 0.3])
% plot(inmcm4_latData,inmcm4_Avg2,'Color',[0.4 0.4 0.4])
% plot(MIROC5_latData,MIROC5_Avg2,'Color',[0.5 0.5 0.5])
% plot(MIROC_ESM_latData,MIROC_ESM_Avg2,'Color',[0.6 0.6 0.6])
% plot(MIROC_ESM_CHEM_latData,MIROC_ESM_CHEM_Avg2,'Color',[0.7 0.7 0.7])
% plot(MRI_CGCM3_latData,MRI_CGCM3_Avg2,'Color',[0.8 0.8 0.8])
% plot(MRI_ESM1_latData,MRI_ESM1_Avg2,'Color',[0.9 0.9 0.9])
% plot(NorESM1_M_latData,NorESM1_M_Avg2,'Color',[0.1 0.2 0.3])
% hold off
% xlim([-90,90])
% ylim([0,1])
% xlabel('Latitude')
% ylabel('Soil moisture component')
% set(gca,'XTick',[-90,-60,-30,0,30,60,90])
% title('1/90<f<1/30 day^{-1}')
% % legend({'SMAP','BCC-CSM1.1','BNU-ESM','CNRM-CM5','CSIRO-Mk3.6','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','HadGEM2-CC','HadGEM2-ES','Institute for Numerical Mathematics','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MRI-CGCM3','MRI-ESM1','NorESM1-M'},'FontSize',4.7)
% subplot(2,2,3)
% plot(latSMAP,SMAP_Avg3,'Color',[0 0 0],'Linewidth',1)
% hold on
% plot(BCC_CSM_latData,BCC_CSM_Avg3,'Color',[0 0 1])
% plot(BNU_ESM_latData,BNU_ESM_Avg3,'Color',[0 1 0])
% plot(CanESM2_latData,CanESM2_Avg3,'Color',[0 1 1])
% plot(CNRM_CM5_latData,CNRM_CM5_Avg3,'Color',[1 0 0])
% plot(CSIRO_Mk_latData,CSIRO_Mk_Avg3,'Color',[1 0 1])
% plot(GFDL_CM3_latData,GFDL_CM3_Avg3,'Color',[1 1 0])
% plot(GFDL_ESM2G_latData,GFDL_ESM2G_Avg3,'Color',[1 1 1])
% plot(GFDL_ESM2M_latData,GFDL_ESM2M_Avg3,'Color',[0.1 0.1 0.1])
% plot(HadGEM2_CC_latData,HadGEM2_CC_Avg3,'Color',[0.2 0.2 0.2])
% plot(HadGEM2_ES_latData,HadGEM2_ES_Avg3,'Color',[0.3 0.3 0.3])
% plot(inmcm4_latData,inmcm4_Avg3,'Color',[0.4 0.4 0.4])
% plot(MIROC5_latData,MIROC5_Avg3,'Color',[0.5 0.5 0.5])
% plot(MIROC_ESM_latData,MIROC_ESM_Avg3,'Color',[0.6 0.6 0.6])
% plot(MIROC_ESM_CHEM_latData,MIROC_ESM_CHEM_Avg3,'Color',[0.7 0.7 0.7])
% plot(MRI_CGCM3_latData,MRI_CGCM3_Avg3,'Color',[0.8 0.8 0.8])
% plot(MRI_ESM1_latData,MRI_ESM1_Avg3,'Color',[0.9 0.9 0.9])
% plot(NorESM1_M_latData,NorESM1_M_Avg3,'Color',[0.1 0.2 0.3])
% hold off
% xlim([-90,90])
% ylim([0,1])
% xlabel('Latitude')
% ylabel('Soil moisture component')
% set(gca,'XTick',[-90,-60,-30,0,30,60,90])
% title('1/365<f<1/90 day^{-1}')
% % legend('SMAP','BCC-CSM1.1','BNU-ESM','CNRM-CM5','CSIRO-Mk3.6','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','HadGEM2-CC','HadGEM2-ES','Institute for Numerical Mathematics','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MRI-CGCM3','MRI-ESM1','NorESM1-M')
% subplot(2,2,4)
% plot(latSMAP,SMAP_Avg4,'Color',[0 0 0],'Linewidth',1)
% hold on
% plot(BCC_CSM_latData,BCC_CSM_Avg4,'Color',[0 0 1])
% plot(BNU_ESM_latData,BNU_ESM_Avg4,'Color',[0 1 0])
% plot(CanESM2_latData,CanESM2_Avg4,'Color',[0 1 1])
% plot(CNRM_CM5_latData,CNRM_CM5_Avg4,'Color',[1 0 0])
% plot(CSIRO_Mk_latData,CSIRO_Mk_Avg4,'Color',[1 0 1])
% plot(GFDL_CM3_latData,GFDL_CM3_Avg4,'Color',[1 1 0])
% plot(GFDL_ESM2G_latData,GFDL_ESM2G_Avg4,'Color',[1 1 1])
% plot(GFDL_ESM2M_latData,GFDL_ESM2M_Avg4,'Color',[0.1 0.1 0.1])
% plot(HadGEM2_CC_latData,HadGEM2_CC_Avg4,'Color',[0.2 0.2 0.2])
% plot(HadGEM2_ES_latData,HadGEM2_ES_Avg4,'Color',[0.3 0.3 0.3])
% plot(inmcm4_latData,inmcm4_Avg4,'Color',[0.4 0.4 0.4])
% plot(MIROC5_latData,MIROC5_Avg4,'Color',[0.5 0.5 0.5])
% plot(MIROC_ESM_latData,MIROC_ESM_Avg4,'Color',[0.6 0.6 0.6])
% plot(MIROC_ESM_CHEM_latData,MIROC_ESM_CHEM_Avg4,'Color',[0.7 0.7 0.7])
% plot(MRI_CGCM3_latData,MRI_CGCM3_Avg4,'Color',[0.8 0.8 0.8])
% plot(MRI_ESM1_latData,MRI_ESM1_Avg4,'Color',[0.9 0.9 0.9])
% plot(NorESM1_M_latData,NorESM1_M_Avg4,'Color',[0.1 0.2 0.3])
% hold off
% xlim([-90,90])
% ylim([0,1])
% xlabel('Latitude')
% ylabel('Soil moisture component')
% set(gca,'XTick',[-90,-60,-30,0,30,60,90])
% title('f<1/365 day^{-1}')
% % legend('SMAP','BCC-CSM1.1','BNU-ESM','CNRM-CM5','CSIRO-Mk3.6','GFDL-CM3','GFDL-ESM2G','GFDL-ESM2M','HadGEM2-CC','HadGEM2-ES','Institute for Numerical Mathematics','MIROC5','MIROC-ESM','MIROC-ESM-CHEM','MRI-CGCM3','MRI-ESM1','NorESM1-M')
% sgtitle('Global soil moisture average energy spectra component with latitude')

% subplot(2,2,1)
% plot(SMAP_Avg1,latSMAP,'Color',[0 0 0])
% hold on
% plot(AVG1_new,latSMAP,'Color',[0 0 1])
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/30<f<1/7 day^{-1}')
% subplot(2,2,2)
% plot(SMAP_Avg2,latSMAP,'Color',[0 0 0])
% hold on
% plot(AVG2_new,latSMAP,'Color',[0 0 1])
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/90<f<1/30 day^{-1}')
% subplot(2,2,3)
% plot(SMAP_Avg3,latSMAP,'Color',[0 0 0])
% hold on
% plot(AVG3_new,latSMAP,'Color',[0 0 1])
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('1/365<f<1/90 day^{-1}')
% subplot(2,2,4)
% plot(SMAP_Avg4,latSMAP,'Color',[0 0 0])
% hold on
% plot(AVG4_new,latSMAP,'Color',[0 0 1])
% hold off
% xlim([0,1])
% ylim([-90,90])
% xlabel('Soil moisture component')
% ylabel('Latitude')
% set(gca,'YTick',[-90,-60,-30,0,30,60,90])
% title('f<1/365 day^{-1}')