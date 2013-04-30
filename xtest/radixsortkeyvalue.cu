#include "common/xtestrunner.h"
#include <algorithm>

#include <xcbb/xcbb.h>

class RadixSortKeyValueTest: public CudaTest
{
protected:
    RadixSortKeyValueTest() {}
    virtual ~RadixSortKeyValueTest() {}
    virtual void SetUp()
    {
        CudaTest::SetUp();
        std::srand(time(0));
    }
    virtual void TearDown() {}

protected:

    template <int RADIX_DIGITS>
    void CreateSample0(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i] = 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample1(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i] = (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample2(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i] = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample3(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample4(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample5(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }

    template <int RADIX_DIGITS>
    void CreateSample6(std::vector<uint>& key, std::vector<uint>& val)
    {
        for (int i = 0; i < key.size(); ++i) key[i]  = (rand() % RADIX_DIGITS) << 0;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 4;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 8;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 12;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 16;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 20;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 24;
        for (int i = 0; i < key.size(); ++i) key[i] += (rand() % RADIX_DIGITS) << 28;
        for (int i = 0; i < key.size(); ++i) val[i] = key[i];
    }
};


/***********************************************************************************
 *
 *
 ***********************************************************************************/
TEST_F(RadixSortKeyValueTest, KeyValue0)
{
    printf("Hello RadixSortKeyValueTest\n");
}




int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
