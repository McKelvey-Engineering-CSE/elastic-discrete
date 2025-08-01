#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <bitset>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>

/************************************************************

Macros directly from NVIDIA themselves for working with 
NVRTC, CUDA, CUDART and JIT calls

*************************************************************/

#define NVRTC_SAFE_CALL(x)                                        \
do {                                                              \
   nvrtcResult result = x;                                        \
   if (result != NVRTC_SUCCESS) {                                 \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
   }                                                              \
} while(0)

#define CUDA_SAFE_CALL(x)                                         \
do {                                                              \
   CUresult result = x;                                           \
   if (result != CUDA_SUCCESS) {                                  \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
   }                                                              \
} while(0)

#define CUDA_NEW_SAFE_CALL(x)                                     \
do {                                                              \
   cudaError_t result = x;                                        \
   if (result != cudaSuccess) {                                   \
      std::cerr << "\nerror: " #x " failed with error "           \
                << cudaGetErrorName(result) << '\n';              \
      exit(1);                                                    \
   }                                                              \
} while(0)

#define NVJITLINK_SAFE_CALL(h,x)                                  \
do {                                                              \
   nvJitLinkResult result = x;                                    \
   if (result != NVJITLINK_SUCCESS) {                             \
      std::cerr << "\nerror: " #x " failed with error "           \
                << result << '\n';                                \
      size_t lsize;                                               \
      result = nvJitLinkGetErrorLogSize(h, &lsize);               \
      if (result == NVJITLINK_SUCCESS && lsize > 0) {             \
         char *log = (char*)malloc(lsize);                        \
         result = nvJitLinkGetErrorLog(h, log);                   \
         if (result == NVJITLINK_SUCCESS) {                       \
            std::cerr << "error: " << log << '\n';                \
            free(log);                                            \
         }                                                        \
      }                                                           \
      exit(1);                                                    \
   }                                                              \
} while(0)

/***********************

simple sm scraper

***********************/

__device__ static unsigned int get_smid() {

  unsigned int r;

  asm("mov.u32 %0, %%smid;" : "=r"(r));

  return r;
}

static __global__ void sm_mapping_scraper(int task_id, int total_tasks, int* task_sms_in_use){

    if (threadIdx.x == 0)
        atomicExch(&task_sms_in_use[(get_smid() * total_tasks) + task_id], 1); 

    __syncthreads();

}

/***********************************

CLI Visualization Functions

************************************/

static std::vector<int> generate_sm_taskset(int* sm_set, int task_count, int sm_num){

    std::vector<int> tasks_on_sm;

    for (int i = 0; i < task_count; i++){
        if (sm_set[(task_count * sm_num) + i])
            tasks_on_sm.push_back(i);
    }
    
    return tasks_on_sm;
}

static int get_bit(int num, int pos){
    return ((num & (1 << pos)) != 0);
}

static std::string map_to_color(int number) {

    int colors[3] = {0};

    //make the base color
    int base_colors = number % 8;
    int wrap_around = std::floor(number / 8);
    for (int i = 0; i < 3; i++)
        colors[i] += 255 * get_bit(base_colors, i);

    //walk the color down
    colors[0] = colors[0] - (60 * wrap_around);
    colors[1] = colors[1] - (60 * wrap_around);
    colors[2] = colors[2] - (60 * wrap_around);

    //make the colors slightly more distinct based on their
    //wrap around value
    for (int i = 0; i < 3; i++){

        if (!get_bit(base_colors - 1, i) != !(wrap_around % 2 == 0)){

            if (wrap_around % 2 == 0)
                colors[i] += 25 * wrap_around;
            else
                colors[i] -= 12.5 * wrap_around;
        }
    }
        

    //resolve any negatives
    if (colors[0] < 0) colors[0] = 0;
    if (colors[1] < 0) colors[1] = 0;
    if (colors[2] < 0) colors[2] = 0;

    //make sure we don't get pure gray
    if (number == 8){
        colors[0] = 25;
        colors[1] = 25;
        colors[2] = 112;
    }

    if (number % 16 == 0){
        colors[0] = 25 * (number / 7);
        colors[2] = 112 * (number / 7);
    }

    // Print the colored text
    std::string color_template("\x1b[38;2;" + std::to_string(colors[0]) + ";" + std::to_string(colors[1]) + ";" + std::to_string(colors[2]) + "m");

    return color_template;

}

static std::string color(std::string combine, std::vector<int> sm_assignment, int* last_color_used = nullptr){

    //in the event that no one owns the sm we are checking, just return it
    if (sm_assignment.size() == 0)
        return "\x1b[38;2;0;0;0m" + combine + "\x1b[0m";
    
    std::string final_string = "";
    int current_task_selection = 0;

    if (last_color_used != nullptr)
        current_task_selection = (*last_color_used != -1) ? *last_color_used : 0;

    for (int i = 0; i < (int) combine.length(); i++){

        //make the needed colors
        final_string += map_to_color(sm_assignment[current_task_selection] + 1) + combine[i] + "\x1b[0m";

        current_task_selection = (current_task_selection + 1) % sm_assignment.size();
    }
    
    if (last_color_used != nullptr)
        *last_color_used = current_task_selection;

    return final_string;

}

static void print_boxes(int num_boxes, int printed_boxes, int* array_of_sms, int task_count, int sm_overhang, int* total_tasks_printed) {

  if (sm_overhang != 0)
    num_boxes -= sm_overhang;

  //keep track of what striping we have done
  int task_striping_last_color[num_boxes] = {-1};
    
  // Print top line
  std::cout << "|  ";
  for (int z = 0; z < num_boxes; z++){
    std::cout << color("+------+ ", generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z)), &task_striping_last_color[z]);
  }

  //check if we need to print task stuff
  if (printed_boxes == 0){
    std::cout << "         Partitions:           |";
  }

  else if (*total_tasks_printed < task_count){
    std::cout << "        |  " << color("Part " + ((*total_tasks_printed < 10) ? "0" + std::to_string(*total_tasks_printed) : std::to_string(*total_tasks_printed)), std::vector<int>({*total_tasks_printed})) << "  |          |";
    *total_tasks_printed += 1;
  }

  else if (*total_tasks_printed == task_count){
    std::cout << "        +-----------+          |";
    *total_tasks_printed += 1;
  }

  else{
    std::cout << "                               |";
  }

  
  std::cout << std::endl;

  // Print middle line and number
  std::cout << "|  ";
  for (int z = 0; z < num_boxes; z++){
    std::cout << color("|", generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z)), &task_striping_last_color[z]);

    //'disable' the sm name if not in use in a partition
    if (generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z)).size() != 0)
        std::cout << "SM: " << (((printed_boxes + z) < 10) ? "0" + std::to_string(printed_boxes+z) : std::to_string(printed_boxes+z));
    else
        std::cout << color("SM: ", generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z))) << color((((printed_boxes + z) < 10) ? "0" + std::to_string(printed_boxes+z) : std::to_string(printed_boxes+z)), generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z)));
    std::cout << color("| ", generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z)), &task_striping_last_color[z]);
  }

  //check if we need to print task stuff
  if (printed_boxes == 0){
    std::cout << "        +-----------+          |";
  }

  else if (*total_tasks_printed < task_count){
    std::cout << "        |  " << color("Part " + ((*total_tasks_printed < 10) ? "0" + std::to_string(*total_tasks_printed) : std::to_string(*total_tasks_printed)), std::vector<int>({*total_tasks_printed})) << "  |          |";
    *total_tasks_printed += 1;
  }

  else if (*total_tasks_printed == task_count){
    std::cout << "        +-----------+          |";
    *total_tasks_printed += 1;
  }

  else{
    std::cout << "                               |";
  }

  std::cout << std::endl;

  // Print bottom of the top line
  std::cout << "|  ";
  for (int z = 0; z < num_boxes; z++){
    std::cout << color("+------+ ", generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z)), &task_striping_last_color[z]);
  }

  //check if we need to print task stuff
  if (*total_tasks_printed < task_count){
    std::cout << "        |  " << color("Part " + ((*total_tasks_printed < 10) ? "0" + std::to_string(*total_tasks_printed) : std::to_string(*total_tasks_printed)), std::vector<int>({*total_tasks_printed})) << "  |          |";
    *total_tasks_printed += 1;
  }
  else{
    std::cout << "                               |";
  }

  std::cout << std::endl;


  // Print middle lines
  for (int y = 0; y < 2; y++){
    std::cout << "|  ";
    for (int z = 0; z < num_boxes; z++){
        std::cout << color("|      | ", generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z)), &task_striping_last_color[z]);
    }

  
    //check if we need to print task stuff
    if (*total_tasks_printed < task_count){
        std::cout << "        |  " << color("Part " + ((*total_tasks_printed < 10) ? "0" + std::to_string(*total_tasks_printed) : std::to_string(*total_tasks_printed)), std::vector<int>({*total_tasks_printed})) << "  |          |";
        *total_tasks_printed += 1;
    }

    else if (*total_tasks_printed == task_count){
      std::cout << "        +-----------+          |";
      *total_tasks_printed += 1;
    }

    else{
        std::cout << "                               |";
    }

    std::cout << std::endl;
    }


  // Print bottom line
  std::cout << "|  ";
  for (int z = 0; z < num_boxes; z++){
    std::cout << color("+------+ ", generate_sm_taskset(array_of_sms, task_count, (printed_boxes + z)), &task_striping_last_color[z]);
  }
  

  //check if we need to print task stuff
  if (*total_tasks_printed < task_count){
    std::cout << "        |  " << color("Part " + ((*total_tasks_printed < 10) ? "0" + std::to_string(*total_tasks_printed) : std::to_string(*total_tasks_printed)), std::vector<int>({*total_tasks_printed})) << "  |          |";
    *total_tasks_printed += 1;
  }

  else if (*total_tasks_printed == task_count){
    std::cout << "        +-----------+          |";
    *total_tasks_printed += 1;
  }

  else{
    std::cout << "                               |";
  }

  std::cout << std::endl;
}

static void start_printing_sms(int num, int* sm_array, int task_num, int device_id) {

    // Find the closest integer to the square root
    int root = std::floor(sqrt(num));

    //print general GPU info
    cudaError_t error;
    cudaDeviceProp properties;

    //ensure we can reach the card
    if ((error = cudaGetDevice(&device_id)) != cudaSuccess){
        printf("CUDA device could not be initialized; returning CUDA error code %d\n", error);
        exit(0);
    }

    //ensure we can get the properties of the card
    if ((error = cudaGetDeviceProperties(&properties, device_id)) != cudaSuccess){
        printf("Could not fetch specifications from CUDA device; returning CUDA error code %d\n", error);
        exit(0);
    }

    printf("\nSM Task Partitioning Visualizer on %s with Compute Capability %d.%d\n", properties.name, properties.major, properties.minor);

    std::cout << "+--" << std::string((8 * root) + (root - 1) + 32, '-') << "+" << std::endl;

    int total_tasks_printed_in_key = 0;
    for (int i = 0; i < (num / root); i++)
        print_boxes(root, (i * root), sm_array, task_num, (((i * root) + root) > num) * (((i * root) + root) - num), &total_tasks_printed_in_key);

    //make sure we printed all the needed parts of the partition key
    if (total_tasks_printed_in_key < task_num){
        for (int i = total_tasks_printed_in_key; i < task_num; i++)
            std::cout << "|" << std::string(55, ' ') << "|  " << color("Part " + ((i < 10) ? "0" + std::to_string(i) : std::to_string(i)), std::vector<int>({i})) << "  |          |" << std::endl;
        std::cout << "|" << std::string(55, ' ') << "+-----------+          |" << std::endl;
    }
    std::cout << "+--" << std::string((8 * root) + (root - 1) + 32, '-') << "+" << std::endl;
 
}

/***************************************************

Templates to create the needed streams and contexts
to visually map the sm units for the different
types of partitioning methods that are available

*****************************************************/

template <typename partition_unit> 
typename std::enable_if<std::is_same<partition_unit, cudaStream_t>::value, std::tuple<std::vector<cudaStream_t>, std::vector<CUcontext>>>::type
build_context_and_stream(partition_unit programming_model_type){

    std::vector<cudaStream_t> streams;
    std::vector<CUcontext> contexts;

    streams = std::vector<cudaStream_t>({programming_model_type});

    //set up the context
    contexts = std::vector<CUcontext>({CUcontext()});
    CUDA_SAFE_CALL(cuStreamGetCtx(programming_model_type, &contexts[0]));
    
    return make_tuple(streams, contexts);
};

template <typename partition_unit> 
typename std::enable_if<std::is_same<partition_unit, CUcontext>::value, std::tuple<std::vector<cudaStream_t>, std::vector<CUcontext>>>::type
build_context_and_stream(partition_unit programming_model_type){

    std::vector<cudaStream_t> streams;
    std::vector<CUcontext> contexts;

    CUcontext old;
    CUDA_SAFE_CALL(cuCtxPopCurrent(&old));

    contexts = std::vector<CUcontext>({programming_model_type});

    CUDA_SAFE_CALL(cuCtxPushCurrent(contexts[0]));

    //set up the stream
    streams = std::vector<cudaStream_t>({cudaStream_t()});
    CUDA_NEW_SAFE_CALL(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));

    CUDA_SAFE_CALL(cuCtxPopCurrent(&old));

    return make_tuple(streams, contexts);
};


template <typename partition_unit> 
typename std::enable_if<std::is_same<partition_unit, CUgreenCtx>::value, std::tuple<std::vector<cudaStream_t>, std::vector<CUcontext>>>::type
build_context_and_stream(partition_unit programming_model_type){

    std::vector<cudaStream_t> streams;
    std::vector<CUcontext> contexts;

    //convert the green context
    contexts = std::vector<CUcontext>({CUcontext()});
    CUDA_SAFE_CALL(cuCtxFromGreenCtx(&contexts[0], programming_model_type));
    
    //set up the stream
    streams = std::vector<cudaStream_t>({cudaStream_t()});
    CUDA_SAFE_CALL(cuGreenCtxStreamCreate(&streams[0], programming_model_type, CU_STREAM_NON_BLOCKING, 0));

    return make_tuple(streams, contexts);
};

template <typename partition_unit> 
typename std::enable_if<std::is_same<partition_unit, std::vector<cudaStream_t>>::value, std::tuple<std::vector<cudaStream_t>, std::vector<CUcontext>>>::type
build_context_and_stream(partition_unit programming_model_type){

    std::vector<cudaStream_t> streams;
    std::vector<CUcontext> contexts;

    streams = std::vector<cudaStream_t>(programming_model_type);

    //build the contexts
    for (int i = 0; i < streams.size(); i++){
        contexts.push_back(CUcontext());
        CUDA_SAFE_CALL(cuStreamGetCtx(streams[i], &contexts[i]));
    }

    return make_tuple(streams, contexts);
};

template <typename partition_unit> 
typename std::enable_if<std::is_same<partition_unit, std::vector<CUcontext>>::value, std::tuple<std::vector<cudaStream_t>, std::vector<CUcontext>>>::type
build_context_and_stream(partition_unit programming_model_type){

    std::vector<cudaStream_t> streams;
    std::vector<CUcontext> contexts;

    CUcontext old;
    CUDA_SAFE_CALL(cuCtxPopCurrent(&old));

    contexts = std::vector<CUcontext>(programming_model_type);

    CUDA_SAFE_CALL(cuCtxPushCurrent(contexts[0]));

    //build the streams
    for (int i = 0; i < contexts.size(); i++){

        CUDA_SAFE_CALL(cuCtxPushCurrent(contexts[0]));

        streams.push_back(cudaStream_t());
        CUDA_NEW_SAFE_CALL(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

        CUDA_SAFE_CALL(cuCtxPopCurrent(&old));
    }

    return make_tuple(streams, contexts);
};

template <typename partition_unit> 
typename std::enable_if<std::is_same<partition_unit, std::vector<CUgreenCtx>>::value, std::tuple<std::vector<cudaStream_t>, std::vector<CUcontext>>>::type
build_context_and_stream(partition_unit programming_model_type){

    std::vector<cudaStream_t> streams;
    std::vector<CUcontext> contexts;

    //temp storage
    std::vector<CUgreenCtx> green_contexts (programming_model_type);

    //convert all the green contexts
    for (int i = 0; i < green_contexts.size(); i++){
        contexts.push_back(CUcontext());
        CUDA_SAFE_CALL(cuCtxFromGreenCtx(&contexts[i], green_contexts[i]));
    }
    
    //make the streams
    for (int i = 0; i < green_contexts.size(); i++){
        streams.push_back(cudaStream_t());
        CUDA_SAFE_CALL(cuGreenCtxStreamCreate(&streams[i], green_contexts[i], CU_STREAM_NON_BLOCKING, 0));
    }

    return make_tuple(streams, contexts);
};

template <typename partition_unit> 
void visualize_sm_partitions(partition_unit programming_model_type, int device = 0){

    std::vector<cudaStream_t> streams;
    std::vector<CUcontext> contexts;

    //call the build
    std::tuple<std::vector<cudaStream_t>, std::vector<CUcontext>> ret_val = build_context_and_stream(programming_model_type);
    streams = std::get<0>(ret_val);
    contexts = std::get<1>(ret_val);

    //now allocate the necessary storage
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device); 
    int sm_map[streams.size() * deviceProp.multiProcessorCount] = {0};
    int* device_sm_map;

    CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&device_sm_map, streams.size() * deviceProp.multiProcessorCount * sizeof(int)));
    CUDA_NEW_SAFE_CALL(cudaMemcpy(device_sm_map, sm_map, streams.size() * deviceProp.multiProcessorCount * sizeof(int), cudaMemcpyHostToDevice));

    //now loop over each context stream set and launch the mapping kernel
    for (int i = 0; i < streams.size(); i++){

        //set context
        CUcontext old;
        CUDA_SAFE_CALL(cuCtxPopCurrent(&old));
        CUDA_SAFE_CALL(cuCtxPushCurrent(contexts[i]));

        //launch with stream
        sm_mapping_scraper<<<deviceProp.multiProcessorCount * 2, 1024, 0, streams[i]>>>(i, streams.size(), device_sm_map);
        CUDA_NEW_SAFE_CALL(cudaDeviceSynchronize());

    }

    //copy data back
    CUDA_NEW_SAFE_CALL(cudaMemcpy(sm_map, device_sm_map, streams.size() * deviceProp.multiProcessorCount * sizeof(int), cudaMemcpyDeviceToHost));

    //display the visualization map
    start_printing_sms(deviceProp.multiProcessorCount, sm_map, streams.size(), device);

}

template <typename T>
key_t nameToKey(T name){
    
    //take in the string and make sure it's less than 10 chars
    if (name.length() > 10){
        std::perror("ERROR: key names must be shorter than 10 characters\n");
        exit(-1);
    }

    std::string bits = "";

    //otherwise, make a number representing the string
    for (size_t i = 0; i < name.length(); i++){

        //if it's a digit
        if (name[i] >= 48 && name[i] <= 57){
            std::bitset<6> binaryRepresentation(name[i] - 48);
            bits += binaryRepresentation.to_string();
        }

        else if (name[i] >= 65 && name[i] <= 90){
            std::bitset<6> binaryRepresentation(name[i] - 55);
            bits += binaryRepresentation.to_string();
        }

        else if (name[i] >= 97 && name[i] <= 122){
            std::bitset<6> binaryRepresentation(name[i] - 61);
            bits += binaryRepresentation.to_string();
        }

        else if (name[i] == 95){
            std::bitset<6> binaryRepresentation(62);
            bits += binaryRepresentation.to_string();
        }

        else{
            printf("ERROR: key names must only contain alphanumeric values and underscores: %s\n", name.c_str());
            exit(-1);
        }

    }

    return (key_t)(std::bitset<64>(bits).to_ullong());
}

static int get_semaphore (std::string key){

    int sem_id;

    sem_id = semget(nameToKey(key), 1, IPC_CREAT | 0666);

    if (sem_id == -1) {
        perror("get_semaphore: semget");
        exit(1);
    }

    return sem_id;
}

static int set_semaphore (int sem_id, int val){

    return semctl(sem_id, 0, SETVAL, val);
}

static void wait_semaphore (int sem_id){

    struct sembuf sem_op;

    sem_op.sem_num  = 0;
    sem_op.sem_op   = 0;
    sem_op.sem_flg = 0;

    if (semop(sem_id, &sem_op, 1) == -1) {
      std::cout << "SEM ERROR: " << strerror(errno) << std::endl;
      exit(1);
    }
}

template <typename partition_unit> 
void visualize_sm_partitions_interprocess(partition_unit programming_model_type, int expected_processes, std::string segment_key, int device = 0){

    std::vector<cudaStream_t> streams;
    std::vector<CUcontext> contexts;

    //for now only allow 8 Character long keys
    if (segment_key.length() > 8){
      std::cout << "visualize_sm_partitions_interprocess: Key too long... only keys of length 8 or shorter are supported right now...." << std::endl;
      exit(1);
    }

    //get device info
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device); 

    //call the build
    std::tuple<std::vector<cudaStream_t>, std::vector<CUcontext>> ret_val = build_context_and_stream(programming_model_type);
    streams = std::get<0>(ret_val);
    contexts = std::get<1>(ret_val);

    //make shared structure needed for sync and communication
    typedef struct {
      int ticket = 0;
      int sem = 0;
    } process_id_structure;

    //let the processes race each other to set up the shared memory segment
    int my_process_id;
    bool leader = true;
    int shmid = shmget(nameToKey(segment_key), sizeof(process_id_structure), 0666 | IPC_CREAT | IPC_EXCL);
    int array_shmid = shmget(nameToKey(segment_key+"_A"), streams.size() * deviceProp.multiProcessorCount * expected_processes * sizeof(int), 0666 | IPC_CREAT);
    int sem_id = get_semaphore(segment_key+"_S");

    if (array_shmid == -1 || sem_id == -1) {
        std::cout << "visualize_sm_partitions_interprocess: Shared memory could not be created... Exiting with error: " << strerror(errno) <<  std::endl;
        exit(1);
    }

    //Check if segment already exists
    if (shmid == -1) {
    
      if (errno == EEXIST){
        leader = false;
        shmid = shmget(nameToKey(segment_key), sizeof(process_id_structure), 0666 | IPC_CREAT);
      }
          
      else {
        std::cout << "visualize_sm_partitions_interprocess: Shared memory could not be created... Exiting with error: " << strerror(errno) <<  std::endl;
        exit(1);
      }
    } 

    if (leader){
      if (set_semaphore(sem_id, 0) == -1) {
        std::cout << "SEM ERROR: " << strerror(errno) << std::endl;
        exit(1);
      }
    }
    
    //attach to shared memory
    void* shmaddr = shmat(shmid, NULL, 0);
    if (shmaddr == (void*) -1) {
      std::cout << "visualize_sm_partitions_interprocess: Shared memory could not be attached... Exiting with error: " << strerror(errno) <<  std::endl;
      exit(1);
    }

    void* arr_addr = shmat(array_shmid, NULL, 0);
    if (arr_addr == (void*) -1) {
      std::cout << "visualize_sm_partitions_interprocess: Shared memory could not be attached... Exiting with error: " << strerror(errno) <<  std::endl;
      exit(1);
    }

    //map the data and try to get our number
    int* shared_array = (int*) arr_addr;
    process_id_structure* process_structure = (process_id_structure*) shmaddr;

    //get our process_id
    wait_semaphore(sem_id);

    my_process_id = process_structure->ticket;
    process_structure->ticket += 1;

    set_semaphore(sem_id, 0);

    while(process_structure->ticket != expected_processes);

    //now allocate the necessary storage
    int sm_map[streams.size() * deviceProp.multiProcessorCount] = {0};
    int* device_sm_map;

    std::cout << "Total SM units: " <<  deviceProp.multiProcessorCount << std::endl;

    CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&device_sm_map, streams.size() * deviceProp.multiProcessorCount * sizeof(int)));
    CUDA_NEW_SAFE_CALL(cudaMemcpy(device_sm_map, sm_map, streams.size() * deviceProp.multiProcessorCount * sizeof(int), cudaMemcpyHostToDevice));

    //now loop over each context stream set and launch the mapping kernel
    for (int i = 0; i < (int) streams.size(); i++){

        //set context
        CUcontext old;
        CUDA_SAFE_CALL(cuCtxPopCurrent(&old));
        CUDA_SAFE_CALL(cuCtxPushCurrent(contexts[i]));

        //launch with stream
        sm_mapping_scraper<<<deviceProp.multiProcessorCount * 4, 1024, 0, streams[i]>>>(i, streams.size(), device_sm_map);
        CUDA_NEW_SAFE_CALL(cudaDeviceSynchronize());

    }

    //copy data back
    CUDA_NEW_SAFE_CALL(cudaMemcpy(sm_map, device_sm_map, streams.size() * deviceProp.multiProcessorCount * sizeof(int), cudaMemcpyDeviceToHost));

    //write results back to array
    wait_semaphore(sem_id);

    process_structure->ticket--;

    //add our results
    for (int i = 0; i < deviceProp.multiProcessorCount; i++){
      for (int j = 0; j < (int) streams.size(); j++){
        shared_array[(my_process_id * streams.size()) + (expected_processes * streams.size() * i) + j] = sm_map[(i * streams.size()) + j];
      }
    }

    set_semaphore(sem_id, 0);

    //display the visualization map
    if (leader) {

      //wait for everyone to be done
      while(process_structure->ticket != 0);

      //show results
      start_printing_sms(deviceProp.multiProcessorCount, shared_array, streams.size() * expected_processes, device);

      //clean up
      shmctl(array_shmid, IPC_RMID, 0);
      shmctl(shmid, IPC_RMID, 0);
      semctl(sem_id, IPC_RMID, 0);
    }
}