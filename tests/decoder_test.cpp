#include "CylinderTag.h"
#include <stdexcept>

static void require(bool ok,const std::string& message) {
    if(!ok) throw std::runtime_error(message);
}
static MarkerInfo observations(const vector<int>& dictionary,int phase,int direction,bool mirror,bool missing=false) {
    MarkerInfo group;
    const float values[8]={1.47,1.54,1.61,1.68,1.68,1.61,1.54,1.47};
    auto midpoint=[&](int symbol) {
        const float root=std::sqrt(1+4*(.11f-.2f*values[symbol]));
        return 50*(1+(symbol>=4?root:-root));
    };
    for(int k=0;k<6;++k) {
        if(missing&&k==2) continue;
        int code=dictionary[(phase+direction*k+12)%12];
        if(mirror) code=63-code;
        if(direction<0) code=(7-code%8)*8+(7-code/8);
        const float left=midpoint(code/8),right=midpoint(code%8);
        const float x=k*9;
        group.cornerLists.push_back({{x,0},{x+6,0},{x+6,right-10},{x,left-10},
                                    {x+6,100},{x,100},{x,left+10},{x+6,right+10}});
        group.feature_center.push_back({x+3,50});
        group.edge_length.push_back(6);
    }
    return group;
}
int main(int argc,char**argv) {
    try {
        require(argc==2,"dictionary argument required");
        std::ifstream in(argv[1]);int n,columns,features;in>>n>>columns>>features;
        require(bool(in)&&columns==12,"load dictionary");
        Mat1i dictionary(n,columns);
        for(int& value:dictionary) in>>value;
        corner_detector detector;
        int cases=0;
        for(int id=0;id<n;++id) for(int phase:{0,4,9})
        for(int direction:{1,-1}) for(bool mirror:{false,true}) for(bool missing:{false,true}) {
            vector<int> row(columns);for(int k=0;k<columns;++k) row[k]=dictionary(id,k);
            auto group=observations(row,phase,direction,mirror,missing);
            detector.featureExtractionRobust(group,group,1);
            vector<MarkerInfo> decoded;
            detector.markerDecoderRobust({group},decoded,dictionary,features);
            require(decoded.size()==1,"one result for exact six-column pattern, ID "+std::to_string(id)+" phase "+std::to_string(phase)+" direction "+std::to_string(direction)+" mirror "+std::to_string(mirror)+" missing "+std::to_string(missing));
            require(decoded[0].markerID==id,"wrong ID");
            require(decoded[0].mirrored==mirror,"wrong mirror flag");
            int offset=0;
            for(size_t k=0;k<decoded[0].featurePos.size();++k) {
                if(missing&&offset==2) ++offset;
                require(decoded[0].featurePos[k]==(phase+direction*offset+12)%12,"wrong cyclic phase");
                ++offset;
            }
            ++cases;
        }
        vector<int> row(columns);for(int k=0;k<columns;++k) row[k]=dictionary(0,k);
        auto group=observations(row,0,1,false);
        detector.featureExtractionRobust(group,group,1);
        group.cornerLists.resize(2);group.feature_center.resize(2);group.edge_length.resize(2);
        vector<MarkerInfo> decoded;
        detector.markerDecoderRobust({group},decoded,dictionary,features);
        require(decoded.empty(),"two columns must not identify a marker");
        Mat blank(80,80,CV_8UC1,Scalar(200));
        CylinderTag tag(argv[1]);
        tag.detectRobust(blank,decoded);
        require(decoded.empty(),"blank must not detect a marker");
        Mat tiny(2,2,CV_8UC1,Scalar(200));
        tag.detectRobust(tiny,decoded);
        require(decoded.empty(),"tiny image must not detect a marker");
        std::cout << cases << " exact dictionary/phase/reflection/missing-column cases passed; short/blank/tiny rejected\n";
    } catch(const std::exception&e) {std::cerr<<e.what()<<'\n';return 1;}
}
