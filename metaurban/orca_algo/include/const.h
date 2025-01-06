#ifndef ORCA_CONST_H
#define ORCA_CONST_H


#define Velocity Point
#define Vector Point


#define CN_EPS 0.00001f
#define CN_INFINITY    1000000000
#define CN_PI_CONSTANT 3.14159265359
#define CN_SQRT_TWO    1.41421356237

#define COMMON_SPEED_BUFF_SIZE 1000
#define SPEED_BUFF_SIZE 250
#define SMALL_SPEED 0.1
#define MISSION_SMALL_SPEED 0.001
#define ECBS_SUBOUT_FACTOR 10.0


//XML tags
#define CNS_TAG_ROOT                    "root"

#define CNS_TAG_AGENTS                  "agents"
#define CNS_TAG_ATTR_NUM            "number"
#define CNS_TAG_ATTR_TYPE           "type"

#define CNS_TAG_DEF_PARAMS              "default_parameters"
#define CNS_TAG_ATTR_SIZE               "size"
#define CNS_TAG_ATTR_MAXSPEED           "movespeed"
#define CNS_TAG_ATTR_AGENTSMAXNUM       "agentsmaxnum"
#define CNS_TAG_ATTR_TIMEBOUNDARY       "timeboundary"
#define CNS_TAG_ATTR_SIGHTRADIUS        "sightradius"
#define CNS_TAG_ATTR_TIMEBOUNDARYOBST   "timeboundaryobst"
#define CNS_TAG_ATTR_REPS               "reps"

#define CNS_TAG_AGENT                   "agent"
#define CNS_TAG_ATTR_ID             "id"
#define CNS_TAG_ATTR_STX            "start.xr"
#define CNS_TAG_ATTR_STY            "start.yr"
#define CNS_TAG_ATTR_STT            "start.theta"
#define CNS_TAG_ATTR_GX             "goal.xr"
#define CNS_TAG_ATTR_GY             "goal.yr"
#define CNS_TAG_ATTR_GT             "goal.theta"


#define CNS_TAG_ALG         "algorithm"
#define CNS_TAG_ST              "searchtype"
#define CNS_TAG_MT              "metrictype"
#define CNS_TAG_BT              "breakingties"
#define CNS_TAG_AS              "allowsqueeze"
#define CNS_TAG_CC              "cutcorners"
#define CNS_TAG_HW              "hweight"
#define CNS_TAG_TS              "timestep"
#define CNS_TAG_DEL             "delta"
#define CNS_TAG_TR              "trigger"
#define CNS_TAG_MN              "mapfnum"

#define CNS_TAG_MAP         "map"
#define CNS_TAG_CELLSIZE    "cellsize"
#define CNS_TAG_WIDTH       "width"
#define CNS_TAG_HEIGHT      "height"
#define CNS_TAG_GRID        "grid"
#define CNS_TAG_ROW         "row"

#define CNS_TAG_OBSTS       "obstacles"
#define CNS_TAG_OBST        "obstacle"
#define CNS_TAG_VERTEX      "vertex"
#define CNS_TAG_ATTR_X      "xr"
#define CNS_TAG_ATTR_Y      "yr"

#define CNS_TAG_LOG         "log"
#define CNS_TAG_SUM             "summary"
#define CNS_TAG_ATTR_SR         "successrate"
#define CNS_TAG_ATTR_RUNTIME    "runtime"
#define CNS_TAG_ATTR_FLOWTIME   "flowtime"
#define CNS_TAG_ATTR_MAKESPAN   "makespan"
#define CNS_TAG_ATTR_COL_AGNT   "collisions"
#define CNS_TAG_ATTR_COL_OBST   "collisionsobst"
#define CNS_TAG_PATH            "path"
#define CNS_TAG_ATTR_PATHFOUND  "pathfound"
#define CNS_TAG_ATTR_STEPS      "steps"
#define CNS_TAG_STEP            "step"


#define CNS_SP_ST_THETA     "thetastar"
#define CNS_SP_ST_DIR       "direct"

#define CNS_AT_ST_ORCA          "orca"
#define CNS_AT_ST_ORCADD        "orca-dd"
#define CNS_AT_ST_ORCAPAR       "orca-par"
#define CNS_AT_ST_ORCAPARECBS   "orca-par-ecbs"
#define CNS_AT_ST_ORCARETURN    "orca-return"

#define CNS_AT_ST_COMMONPOINT   "common-point"
#define CNS_AT_ST_SPEEDBUFFER   "speed-buffer"

#define CNS_SP_MT_DIAG      "diagonal"
#define CNS_SP_MT_MANH      "manhattan"
#define CNS_SP_MT_EUCL      "euclid"
#define CNS_SP_MT_CHEB      "chebyshev"

#define CNS_SUM_SUCCESS_RATE            "success_rate"
#define CNS_SUM_RUN_TIME                "run_time"
#define CNS_SUM_FLOW_TIME               "flow_time"
#define CNS_SUM_MAKESPAN                "makespan"
#define CNS_SUM_COLLISIONS              "collisions"
#define CNS_SUM_COLLISIONS_OBS          "collisions_obs"
#define CNS_SUM_MAPF_MEAN_TIME          "mapf_runtime"
#define CNS_SUM_MAPF_INIT_COUNT         "init_count"
#define CNS_SUM_MAPF_UNITE_COUNT        "unite_count"
#define CNS_SUM_MAPF_UPDATE_COUNT       "update_count"
#define CNS_SUM_MAPF_ECBS_COUNT         "ecbs_count"
#define CNS_SUM_MAPF_PAR_COUNT          "par_count"
#define CNS_SUM_MAPF_SUCCESS_COUNT      "success_count"
#define CNS_SUM_MAPF_UNSUCCESS_COUNT    "unsuccess_count"
#define CNS_SUM_MAPF_FLOWTIME           "mapf_flowtime"


#define CNS_MAPF_COMMON_TIME        "mapf_time"
#define CNS_MAPF_INIT_COUNT         "init_count"
#define CNS_MAPF_UNITE_COUNT        "unite_count"
#define CNS_MAPF_UPDATE_COUNT       "update_count"
#define CNS_MAPF_ECBS_COUNT         "ecbs_count"
#define CNS_MAPF_PAR_COUNT          "par_count"
#define CNS_MAPF_SUCCESS_COUNT      "success_count"
#define CNS_MAPF_UNSUCCESS_COUNT    "unsuccess_count"
#define CNS_MAPF_FLOWTIME           "mapf_flowtime"


#define CN_SP_MT_DIAG   0
#define CN_SP_MT_MANH   1
#define CN_SP_MT_EUCL   2
#define CN_SP_MT_CHEB   3

#define CN_SP_ST_THETA  0
#define CN_SP_ST_DIR    1




// Default values
#define CN_DEFAULT_SIZE 1
#define CN_DEFAULT_MAX_SPEED 1
#define CN_DEFAULT_AGENTS_MAX_NUM 10
#define CN_DEFAULT_TIME_BOUNDARY 5.4
#define CN_DEFAULT_OBS_TIME_BOUNDARY 10
#define CN_DEFAULT_RADIUS_OF_SIGHT 10
#define CN_DEFAULT_METRIC_TYPE CN_SP_MT_EUCL
#define CN_DEFAULT_BREAKINGTIES 0
#define CN_DEFAULT_ALLOWSQUEEZE 0
#define CN_DEFAULT_CUTCORNERS 0
#define CN_DEFAULT_HWEIGHT 1
#define CN_DEFAULT_TIME_STEP 0.25
#define CN_DEFAULT_DELTA 0.1
#define CN_DEFAULT_REPS 0.15
#define CN_DEFAULT_MAPFNUM 3
#define CN_DEFAULT_MAPF_MAXTIME 1000
#define CN_DEFAULT_MAPF_TRIGGER COMMON_POINT
#define CN_DEFAULT_START_THETA  0.0


#define CN_DEFAULT_ST               CN_SP_ST_THETA
#define CNS_DEFAULT_ST              CNS_SP_ST_THETA
#define CNS_DEFAULT_AGENT_TYPE      CNS_AT_ST_ORCA
#define CNS_DEFAULT_MAPF_TRIGGER    CNS_AT_ST_COMMONPOINT


#define CNS_TAG_OPT         "options"
#define CNS_TAG_LOGLVL      "loglevel"
#define CNS_TAG_LOGPATH     "logpath"
#define CNS_TAG_LOGFN       "logfilename"


#define CNS_TAG_MAPFN       "mapfilename"


#define CNS_TAG_LPLEVEL     "lplevel"
#define CNS_TAG_HPLEVEL     "hplevel"
#define CNS_TAG_SECTION     "section"
#define CNS_TAG_LOWLEVEL    "lowlevel"

#define CNS_TAG_OPEN        "open"
#define CNS_TAG_POINT       "node"
#define CNS_TAG_CLOSE       "close"

//XML tags' attributes
#define CNS_TAG_ATTR_NUMOFSTEPS     "numberofsteps"
#define CNS_TAG_ATTR_NODESCREATED   "nodescreated"
#define CNS_TAG_ATTR_LENGTH         "length"
#define CNS_TAG_ATTR_LENGTH_SCALED  "length_scaled"
#define CNS_TAG_ATTR_TIME           "time"

#define CNS_TAG_ATTR_F              "F"
#define CNS_TAG_ATTR_G              "g"
#define CNS_TAG_ATTR_PARX           "parent_x"
#define CNS_TAG_ATTR_PARY           "parent_y"
#define CNS_TAG_ATTR_FINX           "finish.x"
#define CNS_TAG_ATTR_FINY           "finish.y"


//Search Parameters
#define CN_SP_BT_GMIN 0
#define CN_SP_BT_GMAX 1



//Log Configuration
#define CN_LP_LEVEL 0

#define CN_LP_LEVEL_NOPE_VALUE      "0"
#define CN_LP_LEVEL_NOPE_WORD       "nope"
#define CN_LP_LEVEL_TINY_VALUE      "0.5"
#define CN_LP_LEVEL_TINY_WORD       "tiny"
#define CN_LP_LEVEL_SHORT_VALUE     "1"
#define CN_LP_LEVEL_SHORT_WORD      "short"
#define CN_LP_LEVEL_MEDIUM_VALUE    "1.5"
#define CN_LP_LEVEL_MEDIUM_WORD     "medium"
#define CN_LP_LEVEL_FULL_VALUE      "2"
#define CN_LP_LEVEL_FULL_WORD       "full"

#define CN_LP_PATH 1
#define CN_LP_NAME 2


//Grid Cell
#define CN_GC_NOOBS 0
#define CN_GC_OBS   1

//Other
#define CNS_OTHER_PATHSELECTION     "*"
#define CNS_OTHER_MATRIXSEPARATOR   ' '


#endif //ORCA_CONST_H
